from __future__ import annotations


REACT_SYSTEM_PROMPT = """You are a helpful assistant playing a simple version of Minecraft (TextCraft).
You can perform the following actions to interact with the environment:
- craft [target count] [target item] using [count] [item1], [count] [item2], ...
- get [count] [item]
- inventory
For reasoning steps, use: think: [your reasoning]
Output exactly ONE line per turn. Do not output any extra text.
""".strip()


REACT_FEW_SHOT_EXAMPLES = """Here is a demo of how to fetch and craft objects.

Crafting commands:
craft 3 dark oak sign using 6 dark oak planks, 1 stick
craft 4 dark oak planks using 1 dark oak log
craft 1 stick using 1 planks
craft 4 stick using 2 bamboo
craft 4 oak planks using 1 oak log
craft 1 dark oak fence using 2 stick, 4 dark oak planks
craft 1 warped stairs using 6 warped planks
craft 3 oak sign using 6 oak planks, 1 stick

Goal: craft dark oak sign

> think: I should check if I can fetch dark oak sign directly from the environment or the inventory.
OK.

> inventory:
Inventory: [stick] (1) [dark oak planks] (8)

> get dark oak sign
Could not find dark oak sign

> think: I cannot get dark oak sign directly, I need to craft it. From the crafting commands, I can use: craft dark oak sign using 6 dark oak planks, 1 stick. Ingredients needed: 6 dark oak planks, 1 stick. Input assumption: I have all the necessary ingredients in my inventory. Let me verify this first.
OK.

> inventory
Inventory: [stick] (1) [dark oak planks] (8)

> think: I found my ingredients: 6 dark oak planks, 1 stick in my inventory. My assumption is true, I can proceed.
OK.

> craft 1 dark oak sign using 6 dark oak planks, 1 stick
Crafted 1 minecraft:dark_oak_sign

> inventory
Inventory: [dark oak sign] (1)

> think: I now have dark oak sign in my inventory. Task Completed!
OK.

Goal: fetch 2 dark oak logs.

> think: I should check my inventory first, to see if I already have dark oak log. Otherwise, I will directly try to get it from the environment.
OK.

> inventory
Inventory: [stick] (1)

> get 2 dark oak logs.
Got 2 dark oak logs

> inventory
Inventory: [dark oak log] (2) [stick] (1)

> think: I have 2 dark oak logs in my inventory. Task Completed!
OK.

Goal: craft 2 oak planks

> think: I should check if I can fetch oak planks directly from the environment or the inventory.
OK.

> inventory:
Inventory: [stick] (1)

> get 2 oak planks
Could not find 2 oak planks

> think: I cannot get oak planks directly, I need to craft it. From the crafting commands, I can use: craft 4 oak planks using 1 oak log. Ingredients needed: 1 oak log. Input assumption: I have the ingredients needed in my inventory. Let me verify this first.
OK.

> inventory
Inventory: [stick] (1)

> think: My assumption is false. I need to get the ingredients: 1 oak log first.
OK.

> get 1 oak log
Got 1 oak log

> craft 4 oak planks using 1 oak log
Crafted 4 minecraft:oak_planks

> inventory
Inventory: [stick] (1) [oak_planks] (4)

> think: I now have 2 oak planks in my inventory. Task Completed!
OK.
"""


REACT_BASE_USER_PROMPT = (
    "You are given few useful crafting recipes to craft items in Minecraft. "
    'Crafting commands are of the format "craft [target object] using [input ingredients]". '
    'You can either "fetch" an object (ingredients) from the inventory or the environment '
    'or "craft" (target) using any of the crafting commands. '
    "You can use ONLY these crafting commands provided, do not use your own crafting commands. "
    'However, if the crafting command uses a generic ingredient like "planks", you can use '
    'special types of the same ingredient e.g. "dark oak planks" in the command instead. '
    "For any natural language or thoughts, use prefix 'think: '.\n\n"
    + REACT_FEW_SHOT_EXAMPLES
    + "Now here is a different goal. You can use these crafting commands to accomplish the goal. "
    "When you have the desired item in your inventory, think: Task Completed! "
    "If you have tried your best but cannot proceed, think: task failed!\n"
)


def build_react_user_prompt(problem: str, history: str = "") -> str:
    problem = str(problem or "").strip()
    prompt = REACT_BASE_USER_PROMPT + "\n" + problem + "\n>"
    if history:
        prompt += str(history)
    return prompt


def append_react_history(history: str, action: str, observation: str) -> str:
    return (history or "") + f" {str(action or '').strip()}\n{str(observation or '').strip()}\n>"


OUR_AGENT_SYSTEM_PROMPT = REACT_SYSTEM_PROMPT
