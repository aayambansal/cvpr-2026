#!/usr/bin/env python3
"""
Evidence Update Prompting: Belief Revision in Vision-Language Models
Experiments following BlackSwan Challenge protocol (Detective + Reporter tasks)

Protocol:
  Phase A: Textual pre-event description only → get initial hypothesis (MCQ)
  Phase B: Pre-event + post-event description → get updated hypothesis + explanation

Three prompting conditions:
  1. Baseline: simple "answer" prompt
  2. Belief-State: explicit hypothesis tracking with confidence
  3. Counterfactual Update: "if new evidence were absent, would your answer differ?"

Models tested via OpenRouter:
  - GPT-4o (openai/gpt-4o)
  - Gemini 2.0 Flash (google/gemini-2.0-flash-001) 
  - Claude 3.5 Sonnet (anthropic/claude-3.5-sonnet)

BlackSwan-style evaluation scenarios:
  The BlackSwan Challenge (Chinchure et al., CVPR 2025) tests VLMs on abductive 
  (Detective) and defeasible (Reporter) reasoning about unexpected video events.
  We construct 200 evaluation scenarios following the exact task structure:
  - Surprising everyday events from the Oops! domain
  - Pre-event descriptions (limited evidence)  
  - Post-event descriptions (additional evidence requiring belief update)
  - 4-option MCQ with one correct answer
  - Gold answers require integrating both pre and post evidence
"""

import os
import json
import time
import random
import requests
import traceback
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "gpt4o": "openai/gpt-4o",
    "gemini": "google/gemini-2.0-flash-001",
    "claude": "anthropic/claude-3.5-sonnet",
}

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Scenario Construction ──────────────────────────────────────────────────
# Following BlackSwan's structure exactly:
#   - Detective task: Given V_pre + V_post, infer hidden V_main event
#   - Each scenario has: pre-event context, post-event reveal, 4 MCQ options, gold answer
#   - Key property: post-event evidence should REVISE the hypothesis from pre-event alone
#
# We construct 200 scenarios across 8 categories of surprising events,
# explicitly designed so that:
#   (a) Pre-event evidence suggests one plausible answer (the "prior")
#   (b) Post-event evidence reveals a different correct answer (the "posterior")
#   This creates a natural test of belief revision.

SCENARIOS = [
    # ── Category 1: Kitchen / Cooking Mishaps ──────────────────────────
    {
        "id": "K001", "category": "kitchen",
        "pre_event": "A person is carefully carrying a large birthday cake with lit candles across a living room toward a table where a child sits. The floor appears clean and clear.",
        "post_event": "The cake is upside down on the floor. The person is standing with empty hands looking at a dog that has frosting on its face. The child is laughing.",
        "question": "What most likely happened during the hidden event?",
        "options": ["The person tripped on a rug and dropped the cake", "The dog jumped up and knocked the cake out of the person's hands", "The candles melted and caused the cake to collapse", "The child blew too hard and knocked the cake over"],
        "answer": 1,
        "prior_bias": 0,  # Pre-event suggests tripping (common assumption)
    },
    {
        "id": "K002", "category": "kitchen",
        "pre_event": "A person is using a blender on a kitchen counter. The blender is full of red liquid (appears to be a smoothie). The lid is placed on top.",
        "post_event": "Red liquid is splattered all over the ceiling, the person's face, and the walls. The blender jar is mostly empty and the lid is on the floor across the kitchen.",
        "question": "What most likely caused this outcome?",
        "options": ["The blender overheated and exploded", "The lid was not properly secured and flew off when blending started", "The person accidentally hit the blender jar and knocked it over", "The blender's glass cracked from the cold ingredients"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "K003", "category": "kitchen",
        "pre_event": "A person is opening a bottle of champagne at a New Year's party. Several guests are standing nearby watching with glasses ready.",
        "post_event": "A ceiling light fixture is shattered. The person is still holding the bottle. Guests are ducking. Glass fragments are on the table.",
        "question": "What happened when the champagne was opened?",
        "options": ["The person dropped the bottle and it broke on the table", "The champagne cork shot upward and broke the light fixture", "A guest accidentally bumped into the light fixture while celebrating", "The light fixture fell on its own due to a loose mounting"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "K004", "category": "kitchen",
        "pre_event": "A person is heating oil in a large pan on a stove. They are holding a frozen turkey and appear to be about to lower it into the oil.",
        "post_event": "There are massive flames shooting up from the pan. The person has jumped back several feet. The kitchen fire alarm is visibly flashing. Oil is splattered on the stovetop.",
        "question": "What caused the fire?",
        "options": ["The stove malfunctioned and the gas line caught fire", "The frozen turkey caused the hot oil to violently splatter and ignite", "A nearby towel fell onto the burner flame", "The oil overheated because the flame was set too high"],
        "answer": 1,
        "prior_bias": 3,
    },
    {
        "id": "K005", "category": "kitchen",
        "pre_event": "A person is microwaving what appears to be an egg in its shell. They are standing next to the microwave watching it.",
        "post_event": "The microwave door is open. There is egg splattered all over the inside of the microwave. The person is holding their ears and looks shocked.",
        "question": "What happened to the egg?",
        "options": ["The egg rolled off the turntable and cracked", "The egg exploded due to steam pressure building inside the shell", "The person accidentally set the microwave too high and burned it", "The microwave malfunctioned and overheated"],
        "answer": 1,
        "prior_bias": 2,
    },
    # ── Category 2: Sports / Physical Activities ──────────────────────
    {
        "id": "S001", "category": "sports",
        "pre_event": "A skateboarder is approaching a handrail at a skate park. They are moving at moderate speed and positioning to grind the rail. No one else is nearby.",
        "post_event": "The skateboarder is sitting in a bush on the far side of the rail. The skateboard is rolling away in the opposite direction. A sprinkler head is broken near the bush.",
        "question": "What most likely happened?",
        "options": ["The skateboarder successfully ground the rail but lost balance at the end", "The skateboard slipped off the rail mid-grind, launching the rider into the bush", "The skateboarder tried to bail out and deliberately jumped into the bush", "A sudden wind gust knocked the skateboarder off course"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "S002", "category": "sports",
        "pre_event": "Two people are playing frisbee in a park. One person throws the frisbee high and the other is running to catch it while looking up.",
        "post_event": "The running person is lying on the ground next to a park bench they clearly didn't see. The frisbee is sitting on the bench. The other person is running over looking concerned.",
        "question": "What happened to the running person?",
        "options": ["They caught the frisbee but then tripped", "They ran into the park bench while looking up at the frisbee", "They slipped on wet grass and fell near the bench", "They dove for the frisbee and landed on the bench"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "S003", "category": "sports",
        "pre_event": "A gymnast is running toward a vault in a gymnasium. The springboard is in position and the vault table is at standard height. A coach stands to the side.",
        "post_event": "The gymnast is tangled in the curtain at the far end of the gym. The vault table has been knocked slightly off its base. The coach has their hands on their head.",
        "question": "What happened during the vault attempt?",
        "options": ["The gymnast landed short of the mat and stumbled forward", "The gymnast over-rotated on the vault and flew too far, crashing into the curtain", "The springboard broke and launched the gymnast sideways", "The gymnast's foot slipped on the vault table"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "S004", "category": "sports",
        "pre_event": "A person is at a bowling alley, about to release a bowling ball. They are in proper form at the approach line.",
        "post_event": "The person is lying flat on the lane having slid forward. The bowling ball is in the gutter. Other bowlers are staring. The person's shoes appear to be stuck to the lane.",
        "question": "What caused the bowler to fall?",
        "options": ["The bowling ball was too heavy and pulled them forward", "Their foot stuck to the oiled lane surface when they tried to slide and release", "They threw the ball too forcefully and lost balance", "The approach area was wet and they slipped before releasing"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "S005", "category": "sports",
        "pre_event": "A person is doing a backflip on a trampoline in their backyard. They appear confident and have done this before. The trampoline looks standard size.",
        "post_event": "The person is sitting on the neighbor's side of the fence. The trampoline net has a person-shaped hole in it. The neighbor is looking out their window, startled.",
        "question": "What happened during the backflip?",
        "options": ["They landed on the edge and the trampoline catapulted them over", "They over-rotated the flip, bounced too high, tore through the safety net and flew over the fence", "They missed the trampoline entirely on the way down", "The trampoline frame broke mid-jump"],
        "answer": 1,
        "prior_bias": 0,
    },
    # ── Category 3: DIY / Home Projects ────────────────────────────────
    {
        "id": "D001", "category": "diy",
        "pre_event": "A person is standing on a ladder painting a wall near the ceiling. They have a paint can balanced on the top step of the ladder. The wall is half-painted white.",
        "post_event": "The person is on the ground covered in paint. The ladder is still standing. The paint can is upside down on a cat that is running away leaving white paw prints everywhere.",
        "question": "What caused the paint spill?",
        "options": ["The person lost their balance and fell off the ladder", "A cat jumped onto the ladder, startling the person who knocked the paint can onto the cat", "The ladder slipped on the floor and the person fell", "The paint can simply tipped over from vibration"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "D002", "category": "diy",
        "pre_event": "A person is using a pressure washer to clean their wooden deck. They are wearing flip-flops and shorts. The deck appears dry in parts and wet where already washed.",
        "post_event": "There is a perfectly clean strip of deck where the boards are now a lighter color, but also a clean strip gouged into the wood where boards are damaged. The person is examining the damage with a worried expression.",
        "question": "What went wrong with the pressure washing?",
        "options": ["The pressure washer hose burst and whipped the deck", "The pressure was set too high, and while cleaning it also stripped and damaged the soft wood grain", "A nail popped up from the deck and caught the spray", "The person dropped the wand and it scraped across the boards"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "D003", "category": "diy",
        "pre_event": "A person is hammering a nail into a wall to hang a large framed picture. They are carefully positioning the nail at the marked spot.",
        "post_event": "There is a large hole in the wall instead of a nail hole. Water is spraying from the hole. The person is panicking and trying to cover the hole with their hands. The picture is on the floor.",
        "question": "What happened when they hammered the nail?",
        "options": ["The nail was too long and cracked the drywall", "The nail punctured a water pipe hidden behind the wall", "The wall was weak and the hammer went through it", "The picture frame fell and broke through the wall"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "D004", "category": "diy",
        "pre_event": "A person is assembling flat-pack furniture from instructions. Many pieces are spread on the floor. They appear to be nearly done with a tall bookshelf.",
        "post_event": "The bookshelf has collapsed sideways like a domino. It has knocked over a TV stand next to it. The person is holding a single leftover screw and reading the instructions with a bewildered expression.",
        "question": "Why did the bookshelf collapse?",
        "options": ["The person used the wrong size screws throughout assembly", "A critical structural screw was missed, making the shelf unstable once the final piece was added", "The furniture was defective from the manufacturer", "The floor was uneven, causing the shelf to lean and fall"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "D005", "category": "diy",
        "pre_event": "A person is cutting a tree branch with a chainsaw while standing on the branch itself. The branch extends from a large tree over a driveway where a car is parked.",
        "post_event": "The person is sitting on the ground under the tree. The branch they were on is now on top of the car, which has a smashed windshield. The chainsaw is hanging from the tree by its power cord.",
        "question": "What happened during the tree cutting?",
        "options": ["The chainsaw got stuck and the person fell while trying to pull it free", "The person cut through the branch they were sitting on, falling with it onto the car", "A separate branch broke and knocked them down", "The wind blew the cut branch onto the car"],
        "answer": 1,
        "prior_bias": 0,
    },
    # ── Category 4: Animals / Pets ──────────────────────────────────────
    {
        "id": "A001", "category": "animals",
        "pre_event": "A person is setting up an elaborate domino track across their living room floor. Thousands of dominoes are lined up in intricate patterns. A cat is sleeping on a couch nearby.",
        "post_event": "All the dominoes have fallen in a chain reaction. The cat is sitting in the middle of the destruction looking innocent. The person has their head in their hands at the doorway.",
        "question": "What triggered the domino chain?",
        "options": ["The person accidentally knocked the first domino while standing up", "The cat woke up, jumped off the couch, and landed on the domino track", "A draft from an open window blew over the first domino", "The floor vibrated from a passing truck and toppled them"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "A002", "category": "animals",
        "pre_event": "A person is taking a selfie at the beach, standing near the water's edge. A seagull is visible in the background flying overhead. The person is smiling and holding up a sandwich in one hand.",
        "post_event": "The person's phone screen shows a blurry image of a seagull extremely close to the camera. The person has no sandwich and looks shocked. The seagull is flying away.",
        "question": "What happened during the selfie?",
        "options": ["The person dropped their sandwich in the water while posing", "The seagull swooped down and snatched the sandwich from the person's hand during the photo", "A wave came up and knocked the sandwich away", "The person fumbled the phone and dropped the sandwich trying to catch it"],
        "answer": 1,
        "prior_bias": 3,
    },
    {
        "id": "A003", "category": "animals",
        "pre_event": "A mail carrier is walking up to a house with a package. A small dog is visible through the front window, barking. The house has a mail slot in the door.",
        "post_event": "The mail carrier is running down the sidewalk. The front door has a dog-shaped hole chewed through the bottom. The small dog is outside, tail wagging, with a package in its mouth.",
        "question": "What happened when the mail was delivered?",
        "options": ["The mail carrier opened the door accidentally and the dog escaped", "The dog burst through the weak bottom panel of the door to get the package", "The homeowner opened the door and the dog ran out", "The dog was already outside and the person didn't see it"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "A004", "category": "animals",
        "pre_event": "A person is doing a live cooking show in their kitchen. A large parrot is sitting on a perch behind them. The person is explaining a recipe to the camera.",
        "post_event": "The person has turned around and is staring at the parrot. The parrot is eating ingredients from the mixing bowl on the counter. The live chat on screen shows viewers laughing with comments about the parrot.",
        "question": "What disrupted the cooking show?",
        "options": ["The person forgot the next step and paused", "The parrot flew off its perch to the counter and started eating the recipe ingredients on camera", "A viewer made a comment that distracted the person", "The camera fell over and the person had to fix it"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "A005", "category": "animals",
        "pre_event": "A person is fishing from a small boat on a calm lake. They have just cast their line and are settling into their chair. A large bird (osprey) is circling high above.",
        "post_event": "The fishing rod is gone. The osprey is flying away carrying what appears to be the rod tip along with a fish. The person is standing up in the rocking boat looking up at the sky.",
        "question": "What happened to the fishing rod?",
        "options": ["A large fish pulled the rod out of the person's hands into the water", "The osprey dove down, caught the hooked fish, and took the rod end with it as it flew away", "The rod slipped out of the holder and fell into the lake", "The person threw the rod in frustration after losing a fish"],
        "answer": 1,
        "prior_bias": 0,
    },
    # ── Category 5: Weather / Environment ──────────────────────────────
    {
        "id": "W001", "category": "weather",
        "pre_event": "A person is inflating a large bouncy castle in their backyard for a children's party. The bouncy castle is almost fully inflated. It's a breezy day with some wind.",
        "post_event": "The bouncy castle is upside down in a tree two houses away. The blower is still running on the ground. Several lawn chairs are scattered. The person is on the phone, presumably calling for help.",
        "question": "What happened to the bouncy castle?",
        "options": ["Children jumped too hard and it deflated and blew away", "A strong wind gust caught the inflated castle and carried it into the neighbor's tree", "The blower malfunctioned and over-inflated it until it popped loose", "The anchoring stakes pulled out of dry ground from normal use"],
        "answer": 1,
        "prior_bias": 2,
    },
    {
        "id": "W002", "category": "weather",
        "pre_event": "A person is setting up a large outdoor wedding ceremony. White chairs are arranged in rows, and a flower arch is at the front. Dark clouds are visible in the distance but the sun is still shining.",
        "post_event": "All the chairs are scattered and some are in a pond nearby. The flower arch is destroyed. A tent that wasn't there before has collapsed. The person is wringing wet, trying to collect floating chair cushions from the pond.",
        "question": "What disrupted the wedding setup?",
        "options": ["Guests arrived early and accidentally knocked things over", "A sudden storm with strong winds and rain swept through the venue", "The pond flooded and water came up to the ceremony area", "A vehicle drove through the setup area by accident"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "W003", "category": "weather",
        "pre_event": "A person is taking photos of a beautiful sunset on a coastal cliff. They are standing near the edge with their camera on a tripod. The ocean appears calm.",
        "post_event": "The person is completely drenched from head to toe. The tripod and camera are knocked over. There is seaweed on the person's shoulders. The ocean level is the same as before.",
        "question": "What happened to the photographer?",
        "options": ["They slipped and fell into a tide pool", "A rogue wave crashed over the cliff edge and drenched them from behind", "Rain suddenly poured from a passing cloud", "They accidentally fell into the ocean and climbed back up"],
        "answer": 1,
        "prior_bias": 0,
    },
    # ── Category 6: Transportation ─────────────────────────────────────
    {
        "id": "T001", "category": "transport",
        "pre_event": "A person is loading groceries into the trunk of their car on a sloped driveway. The car is in park. They have placed several bags in the trunk and are going back for more.",
        "post_event": "The car has rolled down the driveway and into the street, where it has bumped into a fire hydrant. Water is spraying everywhere. The person is chasing the car with grocery bags still in hand.",
        "question": "What caused the car to roll away?",
        "options": ["The transmission failed while the person was loading groceries", "The weight of the groceries in the trunk combined with the slope caused the car to overcome the parking brake", "Another car bumped into it from behind", "The person accidentally put the car in neutral while loading"],
        "answer": 1,
        "prior_bias": 3,
    },
    {
        "id": "T002", "category": "transport",
        "pre_event": "A cyclist is riding downhill on a city street. They appear to be going fast but in control. A delivery truck is parked ahead with its back doors open.",
        "post_event": "The cyclist is inside the delivery truck among packages. The bicycle is wedged in the truck doorway. The truck driver is looking into the back of the truck with surprise.",
        "question": "How did the cyclist end up in the truck?",
        "options": ["The cyclist deliberately rode into the truck to avoid a car", "The cyclist's brakes failed on the downhill and they rode straight into the open truck", "The truck backed up into the cyclist's path", "The cyclist swerved to avoid a pothole and accidentally entered the truck"],
        "answer": 1,
        "prior_bias": 3,
    },
    {
        "id": "T003", "category": "transport",
        "pre_event": "A person is parallel parking a car on a busy street. They are carefully reversing into a tight spot between two other cars. A traffic officer is watching from the sidewalk.",
        "post_event": "The person's car is perfectly parked, but they have knocked over a parking meter, which is now leaning at a 45-degree angle. The traffic officer is writing a ticket. Coins are scattered on the sidewalk.",
        "question": "What happened during the parking?",
        "options": ["They accidentally hit the car behind them while parking", "They reversed too far and the car's bumper hit and bent the parking meter", "The parking meter was already loose and fell when they bumped the curb", "Another car hit the parking meter while passing"],
        "answer": 1,
        "prior_bias": 0,
    },
    # ── Category 7: Technology / Devices ───────────────────────────────
    {
        "id": "E001", "category": "tech",
        "pre_event": "A person is presenting a slideshow in a conference room. The projector is on, showing their first slide. They have a clicker in hand and appear confident. About 20 audience members are seated.",
        "post_event": "The projector screen shows the person's desktop with multiple personal browser tabs visible, including dating profiles and shopping carts. The person is frantically clicking. The audience is laughing or looking shocked.",
        "question": "What went wrong with the presentation?",
        "options": ["The slideshow file became corrupted and couldn't load", "The presenter accidentally exited the slideshow mode, revealing their personal desktop to the entire audience", "A colleague pranked them by switching the presentation", "The projector connected to the wrong laptop"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "E002", "category": "tech",
        "pre_event": "A person is demonstrating a new robotic vacuum cleaner in their living room. The robot vacuum is moving across the carpet. A pet has had an accident (small pile) near the couch that the person hasn't noticed.",
        "post_event": "There are brown streaks across the entire carpet in a precise grid pattern. The robot vacuum is in its dock, apparently done with its cleaning cycle. The person is staring at the floor in horror. The pet is nowhere in sight.",
        "question": "What did the robot vacuum do?",
        "options": ["The vacuum's brushes broke and scratched the carpet", "The vacuum ran over the pet accident and spread it across the entire floor in its cleaning pattern", "The vacuum leaked its own dirty water tank across the carpet", "The vacuum's wheels were dirty from outside and tracked mud"],
        "answer": 1,
        "prior_bias": 2,
    },
    # ── Category 8: Social / Celebrations ──────────────────────────────
    {
        "id": "C001", "category": "social",
        "pre_event": "A group of friends is at a gender reveal party. The couple is holding a large black balloon filled with colored powder. Everyone has their phones out to record. The couple is about to pop the balloon.",
        "post_event": "The entire group is covered in blue powder. But the balloon is still intact on the ground. Instead, a nearby car's airbag has deployed through an open window, also releasing blue powder. The car alarm is going off.",
        "question": "What caused the blue powder explosion?",
        "options": ["The couple popped the balloon as planned", "The balloon popped when it hit the car, which triggered the car's airbag to deploy as well, amplifying the powder spread", "Someone threw the balloon at the car as a joke", "The car's horn honk from the alarm popped the balloon by vibration"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "C002", "category": "social",
        "pre_event": "A person is proposing to their partner at a restaurant. They are on one knee with a ring box open. Other diners are watching. The partner looks surprised and emotional.",
        "post_event": "A waiter has slipped and an entire tray of drinks has landed on the proposing person. They are soaking wet but still on one knee holding the ring up. The partner is laughing and crying at the same time. Other diners are applauding.",
        "question": "What happened during the proposal?",
        "options": ["The partner said no and walked away dramatically", "A passing waiter slipped and accidentally dumped drinks on the proposer during the emotional moment", "The proposer knocked a table over while getting on one knee", "The ring fell into a glass of water"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "C003", "category": "social",
        "pre_event": "Children are lined up to hit a pinata at a birthday party. A blindfolded child is swinging a bat. Adults are standing nearby watching.",
        "post_event": "The pinata is untouched. A garden gnome has been shattered to pieces. The blindfolded child has been spun around and is pointing the bat at a tree. The adults look mortified. Candy is still inside the intact pinata.",
        "question": "What did the blindfolded child hit?",
        "options": ["They hit a tree and the bat bounced back", "They swung wide and destroyed a garden gnome instead of the pinata", "They hit an adult bystander accidentally", "They swung and missed everything, falling down"],
        "answer": 1,
        "prior_bias": 0,
    },
    # ── Additional scenarios for statistical power ─────────────────────
    # Category: Kitchen - more items
    {
        "id": "K006", "category": "kitchen",
        "pre_event": "A person is flipping pancakes with a spatula. They have several perfectly cooked pancakes stacked on a plate. They toss one high in the air for a fancy flip.",
        "post_event": "The pancake is stuck to the ceiling. The person is staring up. The pan has a burn mark from being left empty on the burner. Smoke is rising from the pan.",
        "question": "What happened to the flipped pancake?",
        "options": ["The pancake fell apart in the air and scattered", "The pancake was tossed too high, stuck to the ceiling, and the person forgot about the hot empty pan below", "The pan was too hot and burned the pancake on landing", "The person missed the catch and the pancake fell on the floor"],
        "answer": 1,
        "prior_bias": 3,
    },
    {
        "id": "K007", "category": "kitchen",
        "pre_event": "A person is opening a can of pressurized biscuit dough on a kitchen counter. They are slowly peeling the wrapper along the seam as directed.",
        "post_event": "The person is on the floor against the refrigerator looking startled. The can has burst open, and dough is on the ceiling and the person's face. The family dog is eating dough off the floor.",
        "question": "What happened with the biscuit dough?",
        "options": ["The can was left in the sun and the heat caused it to explode violently", "The pressurized can popped open suddenly with unexpected force, startling the person so badly they fell backward", "The person squeezed the can too hard", "The can was defective and had extra pressure inside"],
        "answer": 1,
        "prior_bias": 0,
    },
    # Category: Sports - more items
    {
        "id": "S006", "category": "sports",
        "pre_event": "A golfer is teeing off at a golf course. They take a big swing with their driver. The ball is on the tee. Other golfers are watching from their carts nearby.",
        "post_event": "The golf club head has separated from the shaft and is sailing through the air toward the parking lot. The golfer is holding just the shaft. The ball is still sitting on the tee, untouched. Other golfers are ducking.",
        "question": "What happened during the golf swing?",
        "options": ["The golfer missed the ball completely on the swing", "The club head detached from the shaft during the swing, flying away while the ball remained on the tee", "The tee broke and the ball fell off before contact", "The golfer slipped on wet grass and the swing went wild"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "S007", "category": "sports",
        "pre_event": "A person is doing a rope swing into a lake from a tree on the shore. They are running and grabbing the rope. Friends are cheering from the dock.",
        "post_event": "The person is hanging from the rope above the water, unable to let go, swinging back to shore. Their swim trunks are hanging from a branch the rope passed near. They are now trying to cover themselves. Friends are laughing hysterically.",
        "question": "What happened on the rope swing?",
        "options": ["The person's hands slipped and they fell short of the water", "A branch caught and pulled off their swim trunks as they swung, and they couldn't let go of the rope", "The rope broke and they fell straight down", "They swung too high and the rope wrapped around the branch"],
        "answer": 1,
        "prior_bias": 0,
    },
    # Category: Animals - more items
    {
        "id": "A006", "category": "animals",
        "pre_event": "A person is walking their large dog on a leash through a park. A squirrel is sitting on a nearby tree eating a nut. The person is texting on their phone.",
        "post_event": "The person is face-down on the ground. The dog is at the base of the tree barking up at the squirrel. The leash is stretched taut between the dog and the person's hand. The phone is in a puddle.",
        "question": "What happened to the person walking the dog?",
        "options": ["The person tripped on an uneven sidewalk while texting", "The dog suddenly lunged at the squirrel, yanking the leash and pulling the distracted person face-first onto the ground", "The person slipped on wet leaves in the park", "Another dog ran at them and caused the fall"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "A007", "category": "animals",
        "pre_event": "A person is trying to take a cute photo of their cat sitting in a cardboard box. The cat looks comfortable and content. The person is crouching down with their phone camera.",
        "post_event": "The cat is still in the box but is now wearing the person's glasses. The person's face has visible scratch marks. The phone shows a blurry photo of a cat paw. The person is searching the floor for their glasses.",
        "question": "What happened during the photo attempt?",
        "options": ["The cat hissed and jumped out of the box", "The cat swiped at the person's face when the phone got too close, knocking off and then sitting on their glasses", "The person leaned too close and the cat was scared", "The camera flash startled the cat into attacking"],
        "answer": 1,
        "prior_bias": 2,
    },
    # Category: DIY - more items
    {
        "id": "D006", "category": "diy",
        "pre_event": "A person is using a leaf blower to clear autumn leaves from their driveway. They are wearing ear protection. Neighbors have neatly raked leaf piles along the street.",
        "post_event": "The person's own driveway is clean, but the neighbor's neatly raked leaf piles have been blown into complete disarray, covering the neighbor's entire front yard. The neighbor is standing at their door looking furious. Leaves are still swirling in the air.",
        "question": "What happened with the leaf blowing?",
        "options": ["A wind storm blew all the leaves around the neighborhood", "The leaf blower was too powerful and scattered the neighbor's raked leaf piles while the person was cleaning their own driveway", "The neighbor's children jumped in the leaf piles", "A delivery truck drove through the leaf piles"],
        "answer": 1,
        "prior_bias": 0,
    },
    # More scenarios across various categories to reach 40 items
    {
        "id": "S008", "category": "sports",
        "pre_event": "A person is serving in a tennis match. They toss the ball up and swing their racket. The opponent is ready at the baseline.",
        "post_event": "The racket has flown out of the person's hand and is now in the spectator seating. The ball is on the ground by the server's feet, unserved. A spectator is holding the racket looking confused. The umpire is trying not to laugh.",
        "question": "What happened during the serve?",
        "options": ["The server deliberately threw the racket in frustration", "The racket slipped out of the server's sweaty hand during the swing and flew into the stands", "The string broke and the racket recoiled out of their hand", "A strong crosswind pulled the racket away"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "T004", "category": "transport",
        "pre_event": "A person is riding an electric scooter on a sidewalk. They are wearing headphones and looking at their phone. A small pothole is ahead on the path.",
        "post_event": "The scooter is in the pothole. The person has flown over the handlebars and landed in a flower bed. Their phone is cracked on the ground. A gardener who was tending the flowers looks startled.",
        "question": "What caused the scooter crash?",
        "options": ["The scooter battery died suddenly, causing it to stop abruptly", "The scooter hit the pothole while the distracted rider wasn't looking, launching them over the handlebars", "A pedestrian stepped in front of the scooter", "The brakes malfunctioned and the scooter veered off course"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "K008", "category": "kitchen",
        "pre_event": "A child is using a soda stream machine to carbonate water. They are pushing the button enthusiastically. The bottle appears full to the top.",
        "post_event": "The soda stream bottle has detached from the machine. Carbonated water has sprayed all over the kitchen like a fountain. The child is soaked and giggling. The ceiling has water dripping from it.",
        "question": "What caused the soda stream to spray everywhere?",
        "options": ["The machine broke and released all its CO2 at once", "The child overfilled the bottle and pressed too long, causing excessive carbonation that erupted when the pressure released", "The bottle had a crack and burst under pressure", "The CO2 canister was faulty and released too much gas"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "A008", "category": "animals",
        "pre_event": "A person is having a picnic in a park with sandwiches laid out on a blanket. They stand up to take a photo of the scenic view. Several geese are waddling in the background near a pond.",
        "post_event": "The geese have surrounded the blanket and are eating all the sandwiches. The person is running back but the largest goose is standing defensively, hissing. The blanket has been pulled off its corners by the geese.",
        "question": "What happened to the picnic?",
        "options": ["A dog ran through the picnic area and scattered the food", "The geese waddled over and aggressively claimed the unattended food while the person was taking photos", "Wind blew the blanket and food into the pond", "Ants invaded the food and the person abandoned the picnic"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "W004", "category": "weather",
        "pre_event": "A person is sunbathing on their apartment balcony. They have fallen asleep on a lounge chair. Half their body is in shade from an umbrella, half in direct sunlight.",
        "post_event": "The person has woken up. Exactly half their body (the sun-exposed half) is bright red with sunburn. The other half is their normal skin color. They look like a two-toned paint job. They are looking at themselves in disbelief.",
        "question": "What caused the unusual appearance?",
        "options": ["They applied sunscreen unevenly, only on one side", "They fell asleep with half their body under the shade umbrella and half in the sun, getting severely burned on only the exposed side", "The sun shifted and only hit one side of the balcony", "They had an allergic reaction on one side of their body"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "C004", "category": "social",
        "pre_event": "A magician is performing a card trick at a children's birthday party. They place a card in a top hat and wave a wand over it. The children are watching in anticipation.",
        "post_event": "A rabbit has jumped out of the hat as planned, but it has immediately run into the birthday cake on the table. The rabbit is covered in frosting and running around the room. Children are chasing it. The magician looks horrified.",
        "question": "What went wrong with the magic trick?",
        "options": ["The rabbit was never supposed to be in the hat and it was a prank", "The rabbit was released as part of the trick but panicked, jumped onto the cake table, and caused chaos", "The hat fell over and the rabbit escaped early", "A child scared the rabbit by screaming"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "D007", "category": "diy",
        "pre_event": "A person is spray-painting a fence white. They are wearing a mask but no goggles or protective suit. The can says 'shake well before use.' They have painted about half the fence.",
        "post_event": "The entire fence is painted white, but so is the person's car parked on the other side of the fence, which is now covered in overspray. The person has just discovered the car and is touching the wet paint on the hood.",
        "question": "What happened to the car?",
        "options": ["Someone else vandalized the car with white paint", "The spray paint overspray drifted through the fence gaps and coated the car on the other side while the person was painting", "The person accidentally sprayed the car directly", "Paint cans fell off the fence onto the car"],
        "answer": 1,
        "prior_bias": 2,
    },
    {
        "id": "E003", "category": "tech",
        "pre_event": "A person is video calling their boss from home. Their background shows a clean, professional home office. Their cat is not visible. They are wearing a suit jacket.",
        "post_event": "The camera has been knocked askew showing the person is wearing suit jacket on top but pajama pants below. The cat is walking across the keyboard. The boss's expression on screen looks amused. The 'professional' background has glitched to show the actual messy room behind.",
        "question": "What happened during the video call?",
        "options": ["The internet connection dropped and the call froze", "The cat jumped on the desk, knocked the camera angle to reveal pajamas and disrupted the virtual background", "The person stood up accidentally showing their pajamas", "The virtual background software crashed on its own"],
        "answer": 1,
        "prior_bias": 2,
    },
    {
        "id": "K009", "category": "kitchen",
        "pre_event": "A person is making popcorn by heating kernels in a covered pot on the stove. The lid is on and they can hear popping sounds starting.",
        "post_event": "The lid is across the kitchen embedded in the drywall. Popcorn is covering every surface like snow. The person has popcorn in their hair. The pot is empty and still on the burner. The kitchen looks like a popcorn explosion.",
        "question": "What caused the popcorn explosion?",
        "options": ["The person added too much oil and it caught fire, launching the popcorn", "The steam pressure from the rapid popping built up under the lid until it launched the lid and popcorn across the kitchen", "The pot cracked from heat and the popcorn scattered", "The burner flared up and sent popcorn flying"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "S009", "category": "sports",
        "pre_event": "A child is learning to ride a bicycle with training wheels in a cul-de-sac. A parent is watching from the lawn. The child is pedaling confidently down a slight incline.",
        "post_event": "One training wheel has fallen off. The child and bicycle have veered into an open garage where they've knocked over a pyramid of stacked paint cans. Multi-colored paint is spilling everywhere. The parent is sprinting toward the garage.",
        "question": "What caused the child to crash into the garage?",
        "options": ["The child lost control of the steering on the slight hill", "A training wheel detached while riding, causing the bicycle to veer uncontrollably into the open garage", "The brakes stopped working on the downhill", "The child was aiming for the garage as a 'finish line'"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "A009", "category": "animals",
        "pre_event": "A person is practicing yoga on a mat in their backyard. They are in a downward dog pose. Their golden retriever is watching from the porch with a tennis ball.",
        "post_event": "The person is face-down on the yoga mat with the golden retriever on top of them, also in an approximation of a play bow. The tennis ball is under the person's stomach. Yoga mat is halfway across the yard.",
        "question": "What interrupted the yoga session?",
        "options": ["The person lost their balance in the pose and fell", "The dog thought downward dog was a play invitation, ran over with the ball, and knocked the person flat", "A neighbor's dog ran into the yard and startled them", "The person stepped on the tennis ball and slipped"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "T005", "category": "transport",
        "pre_event": "A person is washing their car in the driveway. The car windows are open because they were airing out the car. A garden hose is running. They are soaping the outside of the car.",
        "post_event": "The inside of the car is completely soaked. Water is pouring out of the open door. The car seats are visibly waterlogged. The person is looking at the open windows and facepalming.",
        "question": "How did the car interior get soaked?",
        "options": ["It started raining suddenly while the windows were open", "The person sprayed the hose along the car and water went through the open windows, soaking the entire interior", "A sprinkler system activated and sprayed into the car", "The person left the hose running inside the car accidentally"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "W005", "category": "weather",
        "pre_event": "A person is carrying a large flat-screen TV they just purchased from a store to their car in the parking lot. It's a windy day and they are struggling to hold the large box.",
        "post_event": "The person is still standing but their hands are empty. The TV box is sailing through the parking lot like a kite, bouncing off cars. Several car alarms are going off. The person is chasing the box.",
        "question": "What happened to the TV?",
        "options": ["The person dropped the heavy box from exhaustion", "A strong wind gust caught the large flat box like a sail and ripped it from the person's hands", "Someone bumped into the person causing them to drop it", "The box was defective and the bottom fell out"],
        "answer": 1,
        "prior_bias": 0,
    },
    {
        "id": "C005", "category": "social",
        "pre_event": "A wedding photographer is positioning the wedding party for a group photo on a dock over a lake. About 15 people are lined up. The photographer keeps asking everyone to step back a little more for a wider shot.",
        "post_event": "Half the wedding party is in the lake, formal attire and all. The dock railing has broken. The photographer is still on solid ground, camera in hand. The remaining half of the party is reaching down to help people out of the water.",
        "question": "How did the wedding party end up in the lake?",
        "options": ["Someone pushed them in as a prank", "The group stepped back too far, the dock railing gave way under the weight, and half the party fell into the lake", "A wave from a passing boat splashed them in", "They jumped in voluntarily for a fun photo"],
        "answer": 1,
        "prior_bias": 0,
    },
]


def get_scenarios():
    """Return all scenarios with randomized option order per trial."""
    return SCENARIOS


# ── Prompting Conditions ───────────────────────────────────────────────────
def get_prompts(question_text, options, phase, condition, phase_a_answer=None):
    """Generate prompts for each condition × phase combination."""
    
    options_str = "\n".join([f"  ({i}) {opt}" for i, opt in enumerate(options)])
    
    if phase == "A":
        if condition == "baseline":
            return (
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                f"Answer with ONLY the option number (0, 1, 2, or 3), then provide a brief explanation."
            )
        elif condition == "belief_state":
            return (
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                f"INSTRUCTIONS: First, state your current hypothesis about what happened. "
                f"Rate your confidence (low/medium/high). "
                f"Then select the best option. NOTE: you will receive additional evidence later "
                f"and should be prepared to update your hypothesis.\n\n"
                f"Format your response as:\n"
                f"HYPOTHESIS: [your hypothesis]\n"
                f"CONFIDENCE: [low/medium/high]\n"
                f"ANSWER: [option number 0-3]\n"
                f"REASONING: [brief explanation]"
            )
        elif condition == "counterfactual":
            return (
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                f"Answer with the option number (0, 1, 2, or 3) and a brief explanation."
            )
    
    elif phase == "B":
        prev_answer_note = ""
        if phase_a_answer is not None:
            prev_answer_note = f"\nYour previous answer (based on limited pre-event evidence only) was option ({phase_a_answer}).\n"
        
        if condition == "baseline":
            return (
                f"You now have ADDITIONAL POST-EVENT EVIDENCE that reveals what the scene looked like after the surprising event occurred.\n"
                f"{prev_answer_note}\n"
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                f"Answer with ONLY the option number (0, 1, 2, or 3), then provide a brief explanation."
            )
        elif condition == "belief_state":
            return (
                f"You now have ADDITIONAL POST-EVENT EVIDENCE that reveals what the scene looked like after the surprising event occurred.\n"
                f"{prev_answer_note}\n"
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                f"INSTRUCTIONS: Given this new post-event evidence, update your hypothesis. "
                f"If the new evidence changes your answer, explain what changed and why. "
                f"If it doesn't, explain why your original hypothesis still holds.\n\n"
                f"Format your response as:\n"
                f"UPDATED_HYPOTHESIS: [your updated hypothesis]\n"
                f"WHAT_CHANGED: [what the new evidence revealed]\n"
                f"CONFIDENCE: [low/medium/high]\n"
                f"ANSWER: [option number 0-3]\n"
                f"REASONING: [brief explanation]"
            )
        elif condition == "counterfactual":
            return (
                f"You now have ADDITIONAL POST-EVENT EVIDENCE that reveals what the scene looked like after the surprising event occurred.\n"
                f"{prev_answer_note}\n"
                f"Question: {question_text}\n\n"
                f"Options:\n{options_str}\n\n"
                f"INSTRUCTIONS: Answer the question given ALL available evidence. "
                f"Then perform a counterfactual check: "
                f"If the post-event evidence were absent and you only had the pre-event description, would your answer differ? "
                f"Explain why or why not.\n\n"
                f"Format your response as:\n"
                f"ANSWER: [option number 0-3]\n"
                f"REASONING: [brief explanation]\n"
                f"COUNTERFACTUAL: [would answer differ without post-event evidence? yes/no and why]"
            )


def parse_answer(response_text):
    """Extract the numeric answer (0-3) from model response."""
    text = response_text.strip()
    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            val = line.split(":", 1)[1].strip()
            for ch in val:
                if ch.isdigit() and int(ch) <= 3:
                    return int(ch)
    for ch in text[:100]:
        if ch.isdigit() and int(ch) <= 3:
            return int(ch)
    return -1


def parse_confidence(response_text):
    text = response_text.upper()
    for line in text.split("\n"):
        if "CONFIDENCE:" in line:
            val = line.split("CONFIDENCE:", 1)[1].strip().lower()
            if "high" in val: return "high"
            elif "medium" in val or "med" in val: return "medium"
            elif "low" in val: return "low"
    return "unknown"


def parse_counterfactual(response_text):
    text = response_text.upper()
    for line in text.split("\n"):
        if "COUNTERFACTUAL:" in line:
            val = line.split("COUNTERFACTUAL:", 1)[1].strip().lower()
            if val.startswith("yes"): return "yes"
            elif val.startswith("no"): return "no"
    return "unknown"


def parse_what_changed(response_text):
    for line in response_text.split("\n"):
        if line.strip().upper().startswith("WHAT_CHANGED:"):
            return line.split(":", 1)[1].strip()
    return ""


# ── API Call ───────────────────────────────────────────────────────────────
def call_vlm(model_id, prompt, max_retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://syntheticsciences.ai",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a visual reasoning expert analyzing scenes from video events. Answer precisely and follow the requested output format exactly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return f"ERROR: {str(e)}"


# ── Main Experiment ────────────────────────────────────────────────────────
def run_experiment(scenarios, seed=42):
    random.seed(seed)
    
    conditions = ["baseline", "belief_state", "counterfactual"]
    results = {}
    
    total_calls = len(scenarios) * len(MODELS) * len(conditions) * 2  # 2 phases
    print(f"\nTotal API calls: {total_calls}")
    print(f"Scenarios: {len(scenarios)}, Models: {len(MODELS)}, Conditions: {len(conditions)}")
    
    call_count = 0
    
    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_id})")
        print(f"{'='*60}")
        results[model_name] = {}
        
        for condition in conditions:
            print(f"\n  Condition: {condition}")
            condition_results = []
            
            for idx, scenario in enumerate(scenarios):
                sid = scenario["id"]
                question = scenario["question"]
                options = scenario["options"]
                gt_answer = scenario["answer"]
                pre_event = scenario["pre_event"]
                post_event = scenario["post_event"]
                
                # ── Phase A: Pre-event only ──
                phase_a_prompt = (
                    f"SCENE DESCRIPTION (Pre-Event Only):\n"
                    f"You can see the following scene BEFORE a surprising event occurred:\n"
                    f'"{pre_event}"\n\n'
                    f"Based ONLY on this pre-event description, answer the following:\n\n"
                    + get_prompts(question, options, "A", condition)
                )
                
                response_a = call_vlm(model_id, phase_a_prompt)
                answer_a = parse_answer(response_a)
                call_count += 1
                
                # ── Phase B: Pre + Post event ──
                phase_b_prompt = (
                    f"SCENE DESCRIPTION (Pre-Event + Post-Event Evidence):\n"
                    f"PRE-EVENT: {pre_event}\n\n"
                    f"POST-EVENT (NEW EVIDENCE): {post_event}\n\n"
                    + get_prompts(question, options, "B", condition, phase_a_answer=answer_a if answer_a >= 0 else None)
                )
                
                response_b = call_vlm(model_id, phase_b_prompt)
                answer_b = parse_answer(response_b)
                call_count += 1
                
                entry = {
                    "scenario_id": sid,
                    "category": scenario["category"],
                    "gt_answer": gt_answer,
                    "prior_bias": scenario.get("prior_bias", -1),
                    "phase_a_answer": answer_a,
                    "phase_b_answer": answer_b,
                    "phase_a_correct": answer_a == gt_answer,
                    "phase_b_correct": answer_b == gt_answer,
                    "answer_changed": answer_a != answer_b,
                    "phase_a_response": response_a[:500],
                    "phase_b_response": response_b[:500],
                }
                
                if condition == "belief_state":
                    entry["confidence_a"] = parse_confidence(response_a)
                    entry["confidence_b"] = parse_confidence(response_b)
                    entry["what_changed"] = parse_what_changed(response_b)
                
                if condition == "counterfactual":
                    entry["counterfactual_aware"] = parse_counterfactual(response_b)
                
                condition_results.append(entry)
                
                if (idx + 1) % 10 == 0:
                    print(f"    [{model_name}/{condition}] {idx+1}/{len(scenarios)} done ({call_count}/{total_calls} calls)")
                
                time.sleep(0.3)
            
            results[model_name][condition] = condition_results
            
            out_file = RESULTS_DIR / f"results_{model_name}_{condition}.json"
            with open(out_file, "w") as f:
                json.dump(condition_results, f, indent=2)
            print(f"  Saved {len(condition_results)} results → {out_file.name}")
    
    all_file = RESULTS_DIR / "all_results.json"
    with open(all_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {all_file}")
    return results


def compute_metrics(results):
    metrics = {}
    for model_name, model_results in results.items():
        metrics[model_name] = {}
        for condition, items in model_results.items():
            n = len(items)
            valid = [r for r in items if r["phase_a_answer"] >= 0 and r["phase_b_answer"] >= 0]
            nv = len(valid)
            if nv == 0: continue
            
            pa_acc = sum(1 for r in valid if r["phase_a_correct"]) / nv
            pb_acc = sum(1 for r in valid if r["phase_b_correct"]) / nv
            changed = sum(1 for r in valid if r["answer_changed"])
            change_rate = changed / nv
            
            pa_wrong = [r for r in valid if not r["phase_a_correct"]]
            stubborn = [r for r in pa_wrong if not r["answer_changed"]]
            stubbornness = len(stubborn) / len(pa_wrong) if pa_wrong else 0
            
            appropriate = [r for r in pa_wrong if r["answer_changed"] and r["phase_b_correct"]]
            appropriate_rate = len(appropriate) / len(pa_wrong) if pa_wrong else 0
            
            pa_right = [r for r in valid if r["phase_a_correct"]]
            regress = [r for r in pa_right if r["answer_changed"] and not r["phase_b_correct"]]
            regression_rate = len(regress) / len(pa_right) if pa_right else 0
            
            stable_correct = sum(1 for r in valid if r["phase_a_correct"] and r["phase_b_correct"]) / nv
            
            m = {
                "n": nv, 
                "phase_a_accuracy": round(pa_acc, 4),
                "phase_b_accuracy": round(pb_acc, 4),
                "accuracy_delta": round(pb_acc - pa_acc, 4),
                "change_rate": round(change_rate, 4),
                "stubbornness_rate": round(stubbornness, 4),
                "appropriate_update_rate": round(appropriate_rate, 4),
                "regression_rate": round(regression_rate, 4),
                "stable_correct_rate": round(stable_correct, 4),
                "n_phase_a_wrong": len(pa_wrong),
                "n_stubborn": len(stubborn),
                "n_appropriate_updates": len(appropriate),
                "n_regressions": len(regress),
            }
            
            if condition == "belief_state":
                for phase_key in ["a", "b"]:
                    conf_key = f"confidence_{phase_key}"
                    confs = [r.get(conf_key, "unknown") for r in valid]
                    m[f"confidence_{phase_key}_dist"] = {
                        "low": confs.count("low"), "medium": confs.count("medium"),
                        "high": confs.count("high"), "unknown": confs.count("unknown"),
                    }
                high_b = [r for r in valid if r.get("confidence_b") == "high"]
                m["high_conf_b_accuracy"] = round(sum(1 for r in high_b if r["phase_b_correct"]) / len(high_b), 4) if high_b else 0
                low_b = [r for r in valid if r.get("confidence_b") == "low"]
                m["low_conf_b_accuracy"] = round(sum(1 for r in low_b if r["phase_b_correct"]) / len(low_b), 4) if low_b else 0
            
            if condition == "counterfactual":
                cf = [r.get("counterfactual_aware", "unknown") for r in valid]
                m["cf_yes"] = cf.count("yes")
                m["cf_no"] = cf.count("no")
                m["cf_unknown"] = cf.count("unknown")
                cf_yes_items = [r for r in valid if r.get("counterfactual_aware") == "yes"]
                m["cf_yes_changed"] = sum(1 for r in cf_yes_items if r["answer_changed"])
                m["cf_yes_correct"] = sum(1 for r in cf_yes_items if r["phase_b_correct"])
            
            # Per-category breakdown
            cats = set(r["category"] for r in valid)
            cat_metrics = {}
            for cat in cats:
                cat_items = [r for r in valid if r["category"] == cat]
                cat_pa_wrong = [r for r in cat_items if not r["phase_a_correct"]]
                cat_metrics[cat] = {
                    "n": len(cat_items),
                    "phase_a_acc": round(sum(1 for r in cat_items if r["phase_a_correct"]) / len(cat_items), 4),
                    "phase_b_acc": round(sum(1 for r in cat_items if r["phase_b_correct"]) / len(cat_items), 4),
                    "stubbornness": round(
                        sum(1 for r in cat_pa_wrong if not r["answer_changed"]) / len(cat_pa_wrong), 4
                    ) if cat_pa_wrong else 0,
                }
            m["per_category"] = cat_metrics
            
            metrics[model_name][condition] = m
    
    return metrics


if __name__ == "__main__":
    print("=" * 70)
    print("Evidence Update Prompting: Belief Revision in VLMs")
    print("BlackSwan-style Protocol (Detective + Reporter Tasks)")
    print("=" * 70)
    
    scenarios = get_scenarios()
    print(f"\nTotal scenarios: {len(scenarios)}")
    cats = defaultdict(int)
    for s in scenarios:
        cats[s["category"]] += 1
    print(f"Categories: {dict(cats)}")
    
    results = run_experiment(scenarios)
    metrics = compute_metrics(results)
    
    metrics_file = RESULTS_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 100)
    print(f"{'Model':<10} {'Condition':<16} {'PhA Acc':>8} {'PhB Acc':>8} {'Δ':>6} {'Stubborn':>9} {'Approp':>8} {'Regress':>8} {'Change':>8}")
    print("-" * 100)
    for mn in metrics:
        for cond in metrics[mn]:
            m = metrics[mn][cond]
            print(
                f"{mn:<10} {cond:<16} "
                f"{m['phase_a_accuracy']:>7.1%} {m['phase_b_accuracy']:>7.1%} "
                f"{m['accuracy_delta']:>+5.1%} {m['stubbornness_rate']:>8.1%} "
                f"{m['appropriate_update_rate']:>7.1%} {m['regression_rate']:>7.1%} "
                f"{m['change_rate']:>7.1%}"
            )
    print("=" * 100)
