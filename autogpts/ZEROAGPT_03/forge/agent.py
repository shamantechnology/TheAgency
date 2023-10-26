import json
import pprint
import os
import shutil
import time
from datetime import datetime

from forge.sdk import (
    Agent,
    AgentDB,
    Step,
    StepRequestBody,
    Workspace,
    ForgeLogger,
    Task,
    TaskRequestBody,
    PromptEngine,
    chat_completion_request,
    ProfileGenerator,
    num_tokens_from_messages
)

from forge.sdk.memory.memstore import ChromaMemStore
from forge.sdk.memory.memstore_tools import add_chat_memory

from forge.sdk.ai_planning import AIPlanning

LOG = ForgeLogger(__name__)


class ForgeAgent(Agent):
    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)
        
        # initialize chat history
        self.chat_history = []

        # steps completed histroy
        self.steps_completed = []

        # expert profile
        self.expert_profile = None

        # setup chatcompletion to achieve this task
        # with custom prompt that will generate steps
        self.prompt_engine = PromptEngine(os.getenv("OPENAI_MODEL"))

        # ai plan
        self.ai_plan = None
        self.plan_steps = None

        # instruction messages
        self.instruction_msgs = []

        # keep number of steps per task
        self.task_steps_amount = 0

        # track amount of tokens to add a wait
        self.total_token_amount = 0

        # track amount it repeats
        self.repeating = 0

        # memstore
        self.memstore_db = os.getenv("VECTOR_DB")
        self.memstore = None
    
    def copy_to_temp(self, task_id: str) -> None:
        """
        Copy files created from cwd to temp
        This was a temp fix due to the files not being copied but maybe
        fixed in newer forge version
        """
        cwd = self.workspace.get_cwd_path(task_id)
        tmp = self.workspace.get_temp_path()

        for filename in os.listdir(cwd):
            if ".sqlite3" not in filename:
                file_path = os.path.join(cwd, filename)
                if os.path.isfile(file_path):
                    LOG.info(f"copying {str(file_path)} to {tmp}")
                    shutil.copy(file_path, tmp)

    def clear_temp(self) -> None:
        """
        Clear temp folder at each new task
        """
        tmp = self.workspace.get_temp_path()
        LOG.info(f"Clearing temp_folder")
        for filename in os.listdir(tmp):
            file_path = os.path.join(tmp, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                LOG.info(f"Deleted {filename}")
            except Exception as e:
                LOG.error('Failed to delete %s. Reason: %s' % (file_path, e))
    
    async def clear_chat(self, task_id: str, fresh: bool = False) -> None:
        """
        Clear chat and remake with instruction messages
        """
        LOG.info(f"CHAT DUMP\n\n{self.chat_history}\n\n")

        # reset steps
        self.task_steps_amount = 0

        # clear chat and rebuild
        self.chat_history = []
        self.instruction_msgs = []
        self.steps_completed = []

        await self.set_instruction_messages(task_id)

        if not fresh:
            # get last assistant thoughts
            last_thoughts = ""
            last_ability = ""

            # get last/most recent thought
            for i in reversed(range(len(self.chat_history))):
                if self.chat_history[i]["role"] == "assistant":
                    ch_json = json.loads(self.chat_history[i]["content"])
                    last_thoughts = json.dumps(ch_json["thoughts"])
                    break
            
            # get last/most recent ability used
            for i in reversed(range(len(self.chat_history))):
                if self.chat_history[i]["role"] == "function":       
                    last_ability = self.chat_history[i]["content"]
                    break

            self.chat_history.append({
                "role": "user",
                "content": f"Error caused chat reset. Your last thought was\n'{last_thoughts}'\nYour last ability used was\n'{last_ability}'"
            })

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        # set instruction amount to 0
        self.instruct_amt = 0

        # create task
        try:
            task = await self.db.create_task(
                input=task_request.input,
                additional_input=task_request.additional_input
            )

            LOG.info(
                f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
            )
        except Exception as err:
            LOG.error(f"create_task failed: {err}")
        
        # initalize memstore for task
        try:
            if self.memstore_db == "chroma":
                # initialize chroma memstore
                cwd = self.workspace.get_cwd_path(task.task_id)
                chroma_dir = f"{cwd}/chromadb"
                os.makedirs(chroma_dir)
                self.memstore = ChromaMemStore(chroma_dir+"/")
                LOG.info(f"🧠 Created Chroma memorystore @ {os.getenv('AGENT_WORKSPACE')}/{task.task_id}/chroma")
            elif self.memstore_db == "weaviate":
                # initialize weaviate
                LOG.info("🧠 Initializing Weaviate")
                self.memstore = WeaviateMemstore(use_embedded=True)
        except Exception as err:
            LOG.error(f"memstore creation failed: {err}")

        # clear chat and ability history
        self.chat_history = []

        # clear temp folder
        self.clear_temp()
        
        # get role for task
        # use AI to get the proper role experts for the task
        profile_gen = ProfileGenerator(
            task=task
        )

        # keep trying until json loads works, meaning
        # properly formatted reply
        # while solution breaks autogpt
        LOG.info("💡 Generating expert profile...")
        while self.expert_profile is None:
            role_reply = await profile_gen.role_find()
            try:        
                self.expert_profile = json.loads(role_reply)
            except Exception as err:
                pass
                # LOG.error(f"role_reply failed\n{err}")
        LOG.info("💡 Profile generated!")
        # add system prompts to chat for task
        self.instruction_msgs = []
        await self.set_instruction_messages(task.task_id)

        self.task_steps_amount = 0

        return task
    
    async def add_chat(self, 
        task_id: str, 
        role: str, 
        content: str,
        is_function: bool = False,
        function_name: str = None):
        
        if is_function:
            chat_struct = {
                "role": role,
                "name": function_name,
                "content": content
            }
        else:
            chat_struct = {
                "role": role, 
                "content": content
            }
        
        try:
            # cut down on messages being repeated in chat
            if chat_struct not in self.chat_history:
                self.chat_history.append(chat_struct)
            else:
                # clear after three times
                if self.repeating == 3:
                    # clear chat, resend instructions, dont include past messages
                    LOG.info("Stuck in a repeat loop. Waiting 30s then clearing and resetting chat")
                    self.repeating = 0
                    await self.clear_chat(task_id, True)
                else:
                    self.repeating += 1

        except KeyError:
            self.chat_history = [chat_struct]

    async def set_instruction_messages(
        self,
        task_id: str
    ) -> None:
        """
        Add the call to action and response formatting
        system and user messages
        """
        task = await self.db.get_task(task_id)

        # add system prompts to chat for task
        # set up reply json with alternative created
        system_prompt = self.prompt_engine.load_prompt("system-reformat")

        # add to messages
        # wont memory store this as static
        # LOG.info(f"🖥️  {system_prompt}")
        self.instruction_msgs.append(("system", system_prompt))
        await self.add_chat(task_id, "system", system_prompt)

        # add abilities prompt
        abilities_prompt = self.prompt_engine.load_prompt(
            "abilities-list",
            **{"abilities": self.abilities.list_abilities_for_prompt()}
        )

        # LOG.info(f"🖥️  {abilities_prompt}")
        self.instruction_msgs.append(("system", abilities_prompt))
        await self.add_chat(task_id, "system", abilities_prompt)

        # add role system prompt
        try:
            role_prompt_params = {
                "name": self.expert_profile["name"],
                "expertise": self.expert_profile["expertise"]
            }
        except Exception as err:
            LOG.error(f"""
                Error generating role, using default\n
                Name: Joe Anybody\n
                Expertise: Project Manager\n
                err: {err}""")
            role_prompt_params = {
                "name": "Joe Anybody",
                "expertise": "Project Manager"
            }
            
        role_prompt = self.prompt_engine.load_prompt(
            "role-statement",
            **role_prompt_params
        )

        # LOG.info(f"🖥️  {role_prompt}")
        self.instruction_msgs.append(("system", role_prompt))
        await self.add_chat(task_id, "system", role_prompt)

        # setup call to action (cta) with task and abilities
        # use ai to plan the steps
        if not skip_plans:
            self.ai_plan = AIPlanning(
                task=task.input,
                task_id=task_id,
                abilities=self.abilities.list_abilities_for_prompt(),
                workspace=self.workspace
            )

            # plan_steps = None
            # while plan_steps_prompt is None:
            LOG.info("💡 Generating step plans...")
            try:
                self.plan_steps = await self.ai_plan.create_steps()
            except Exception as err:
                # pass
                LOG.error(f"plan_steps_prompt failed\n{err}")
            
            LOG.info(f"🖥️ planned steps\n{self.plan_steps}")
        
        ctoa_prompt_params = {
            "plan": self.plan_steps,
            "task": task.input
        }

        task_prompt = self.prompt_engine.load_prompt(
            "step-work3",
            **ctoa_prompt_params
        )

        # LOG.info(f"🤓 {task_prompt}")
        self.instruction_msgs.append(("user", task_prompt))
        await self.add_chat(task_id, "user", task_prompt)
        # ----------------------------------------------------
    
    async def handle_tokens(self, task_id) -> None:
        """
        Token check to see if reaching limits or need to slow down
        """
        # get amount of tokens for chat with next message
        self.total_token_amount = num_tokens_from_messages(self.chat_history, os.getenv("OPENAI_MODEL"))

        # check if usage token is greater than 1000 and wait
        # this is for the gpt4 api
        LOG.info(f"Total Token Amount: {self.total_token_amount}")
        if os.getenv("OPENAI_MODEL") == "gpt-4":
            if self.total_token_amount >= 8192:
                LOG.info("Token limit of 8192 being reached. Resetting chat")
                await self.clear_chat(task_id)
            elif self.total_token_amount >= 5000:
                LOG.info("Token amount high. Waiting 10secs to not hit limits")
                time.sleep(10)
                LOG.info("Continuing...")
        elif os.getenv("OPENAI_MODEL") == "gpt-3.5-turbo":
            if self.total_token_amount >= 4096:
                LOG.info("Token limit of 4096 being reached. Resetting chat")
                await self.clear_chat(task_id)
        elif os.getenv("OPENAI_MODEL") == "gpt-3.5-turbo-16k":
            if self.total_token_amount >= 16384:
                LOG.info("Token limit of 16384 being reached. Resetting chat")
                await self.clear_chat(task_id)


    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        # task = await self.db.get_task(task_id)

        # Create a new step in the database
        # have AI determine last step
        step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            additional_input=step_request.additional_input,
            is_last=False
        )

        step.status = "running"

        self.task_steps_amount += 1
        LOG.info(f"Step {self.task_steps_amount}")

        # check tokens before chat completion
        await self.handle_tokens(task_id)

        # used in some chat messages
        timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        
        # load current chat into chat completion
        try:
            chat_completion_parms = {
                "messages": self.chat_history,
                "model": os.getenv("OPENAI_MODEL"),
                "temperature": 0.5
            }

            chat_response = await chat_completion_request(
                **chat_completion_parms)
        except Exception as err:
            LOG.error(f"[{timestamp}] llm error")
            # will have to cut down chat or clear it out
            LOG.error("Clearning chat and resending instructions due to API error")
            LOG.error(f"{err}")
            if "RateLimitError" in str(err):
                LOG.error("RATE LIMIT ERROR")
            LOG.error(f"last chat {self.chat_history[-1]}")
            await self.clear_chat(task_id)
        else:
            try:
                answer = json.loads(
                    chat_response["choices"][0]["message"]["content"])
                
                # add response to chat log
                # LOG.info(f"⚙️ chat_response\n{chat_response}")
                await self.add_chat(
                    task_id,
                    "assistant",
                    chat_response["choices"][0]["message"]["content"]
                )
                
                output = None
                
                # make sure about reply format
                if ("ability" not in answer 
                        or "thoughts" not in answer):
                    # LOG.info("Clearning chat and resending instructions due to error")
                    # await self.clear_chat(task_id)
                    LOG.error(f"Answer in wrong format\n\n{answer}\n\n")
                    await self.add_chat(
                        task_id=task_id,
                        role="user",
                        content=f"[{timestamp}] Your reply was not in the correct JSON format."
                    )
                else:
                    # Set the step output and is_last from AI
                    if "speak" in answer["thoughts"]:
                        step.output = answer["thoughts"]["speak"]
                    else:
                        step.output = "Nothing to say..."

                    # get the step completed
                    if "step completed" in answer["thoughts"]:
                        step_completed = answer["thoughts"]["step completed"]
                        if step_completed != "None" and step_completed != None:
                            self.steps_completed.append(answer["thoughts"]["step completed"])
                        
                        LOG.info(f"Steps completed: {self.steps_completed}")
                        
                    LOG.info(f"🤖 {step.output}")

                    if "ability" in answer:

                        LOG.info(f"🤖 {answer['ability']}")

                        # Extract the ability from the answer
                        ability = answer["ability"]

                        if (ability is not None and 
                            ability != "" and 
                            ability != "None"):
                            if (ability["name"] != "" and
                            ability["name"] != None and
                            ability["name"] != "None"):
                                LOG.info(f"🔨 Running Ability {ability}")

            # Run the ability and get the output
            if "args" in ability:
                output = await self.abilities.run_ability(
                    task_id,
                    ability["name"],
                    **ability["args"]
                )
            else:
                output = await self.abilities.run_ability(
                    task_id,
                    ability["name"]
                )

            output = str(output) if output else "None"

                                    # add to converstion
                                    # add arguments to function content, if any
                                    if "args" in ability:
                                        ccontent = f"[Arguments {ability['args']}]: {output} "
                                    else:
                                        ccontent = output

                                    await self.add_chat(
                                        task_id=task_id,
                                        role="function",
                                        content=ccontent,
                                        is_function=True,
                                        function_name=ability["name"]
                                    )

            # Set the step output and is_last from AI
            step.output = answer["thoughts"]["speak"]
            step.is_last = answer["thoughts"]["last_step"]

            except json.JSONDecodeError as e:
                # Handle JSON decoding errors
                # notice when AI does this once it starts doing it repeatingly
                LOG.error(f"agent.py - JSON error, ignoring response: {e}")
                LOG.error(f"🤖 {chat_response['choices'][0]['message']['content']}")

                LOG.info("Clearning chat and resending instructions due to JSON error")
                await self.clear_chat(task_id, True, True)
            except Exception as e:
                # Handle other exceptions
                LOG.error(f"execute_step error: {e}")

                # LOG.info("Clearning chat and resending instructions due to error")
                # await self.clear_chat(task_id)

        # dump whole chat log at last step
        if step.is_last and self.chat_history:
            LOG.info(f"{pprint.pformat(self.chat_history, indent=4, width=100)}")

        # Return the completed step
        return step
