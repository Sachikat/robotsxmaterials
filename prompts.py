prompts = {
    "prompt1": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Your task is to analyze short video clips and their audio and extract the lowest-level human actions being performed.

        Your goal is to extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.

        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. Your task is to take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow.
        Your goal is to return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.
        """ 
    },
    "prompt2": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Your task is to analyze short video clips and their audio and extract the lowest-level human actions being performed.

        Your goal is to extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.
        
        The input of these clips are short video clips and their transcriptions (audio). 

        The output of this task should be a valid json file of the actions done in the video. Do not include explanations, markdown, or commentary.

        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. Your task is to take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow.
        
        Your goal is to return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.

        The input of this task is a JSON file or a list of actions performed in each short video clip. 

        The output of this task should be a mermaid flowchart of the overall task graph that represents the entire procedural workflow. 
        """        
    }, 
    "prompt3": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Your task is to analyze short video clips and their audio and extract the lowest-level human actions being performed.

        Your goal is to extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.

        The input of these clips are short video clips and their transcriptions (audio). 

        The output of this task should be a valid json file of the actions done in the video. Do not include explanations, markdown, or commentary.

        Guidelines:
        - List actions in the order they occur.
        - Include tools, chemicals, and measurements when visible or stated.
        - Do NOT infer intent or future steps beyond what is visible or stated.
        - Do not hallucinate actions when none are visible or stated.


        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. Your task is to take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow.

        Your goal is to return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.

        The input of this task is a JSON file or a list of actions performed in each short video clip. 

        The output of this task should be a mermaid flowchart of the overall task graph that represents the entire procedural workflow. 

        Guidelines: 
        - Respect the order of steps within each clip, and then across clips.
        - Do not hallucinate actions when none are stated.
        - Do not invent steps.
        - Do not change the order of steps.
        """  
    }, 
    "prompt4": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Your task is to analyze short video clips and their audio and extract the lowest-level human actions being performed.

        Goal of this task: 
        Extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.
        For example, you should be extracting low level actions like pick up tweezers, drop tweezers, grab vial from shelf, place hand on cap of **substance name**, twist off cap, etc. 
        The actions should be low-level. They should each logical step the human is performing in the video. 
        The goal is to find all low-level actions in the video clip (not broad/high level actions).
        Use video as the primary source of truth.
        Use audio as a guide to deduce information about the procedure in the clip. The human is voicing their procedure, so use this to achieve the goal. 

        The input of these clips are short video clips and their transcriptions (audio). 

        The output of this task should be a valid json file of the actions done in the video. Do not include explanations, markdown, or commentary.

        Guidelines:
        - List actions in the order they occur.
        - Include tools, chemicals, and measurements when visible or stated.
        - Do NOT infer intent or future steps beyond what is visible or stated.
        - Do not hallucinate actions when none are visible or stated.

        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. Your task is to take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow.

        Goal of this task:
        Return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.
        Based on JSON/list of actions given from each video clip (all these clips make up the procedure), create a mermaid graph to represent a logical task graph of the entire procedural flow. 
        Construct a clean task graph showing the entire procedural flow.
        Use standard nodes, like A, B, C... for steps and use arrows to connect one low level action/step to the next, building the entire procedure.

        The input of this task is a JSON file or a list of actions performed in each short video clip. 

        The output of this task should be a mermaid flowchart of the overall task graph that represents the entire procedural workflow. 

        Guidelines: 
        - Respect the order of steps within each clip, and then across clips.
        - Do not hallucinate actions when none are stated.
        - Do not invent steps.
        - Do not change the order of steps.
        """  
    }, 
    "prompt5": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Task of this prompt:
        Analyze short video clips and their audio.
        Extract the lowest-level human actions being performed.
        Each clip will have actions being performed to achieve a step in the procedure (there may be one action to multiple actions).
        From each clip, get all the low-level actions a human is performing based on what is seen in the video and heard from the audio (or read from the transcription of the audio). 
        Examples of low level actions: pick up flask, pour substance A to 10 mL, set flask down, twist cap, etc. 
        Extraction of actions should only come from the video and audio, do not make up any actions to logically fill in gaps. Do not hallucinate low-level actions from the video clip and audio. 
        Use audio transcription and video clip to see low level actions and what materials/substances are being used.
        Notice physical actions performed by the human, chemicals, tools, and containers used, any measurements or quantities involved

        Your goal is to extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.

        The input of these clips are short video clips and their transcriptions (audio). 

        The output of this task should be a valid json file of the actions done in the video. Do not include explanations, markdown, or commentary.

        Guidelines:
        - List actions in the order they occur.
        - Include tools, chemicals, and measurements when visible or stated.
        - Do NOT infer intent or future steps beyond what is visible or stated.
        - Do not hallucinate actions when none are visible or stated.


        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. 
        Task of this prompt: 
        Take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow. 
        Actions are found within each file. Each file represents a clip of the full procedural video, so action order should be preserved from file to file and within the file. 
        For example, a file may have the first action as picking up a paper, drop paper, pick up flask. These actions should stay in order and no new actions should be added. The next file might be place flask on weighing machine, add substance, etc. The previous action from the previous file (pick up flask) should connect to the next file’s first action (place flask on weighing machine). 
        Doing this, it should create an overall task graph of the entire procedure.
        Given all the list of actions (in list form or JSON), make a task graph going from one step to the other. 
        Order should be preserved within a video clip’s actions (so within the file, the order of the actions should stay the same).
        Order from one list of actions to the next should stay the same (do not hallucinate and create actions to fill in gaps from one list of actions to the next). 

        Your goal is to return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.

        The input of this task is a JSON file or a list of actions performed in each short video clip. 

        The output of this task should be a mermaid flowchart of the overall task graph that represents the entire procedural workflow. 

        Guidelines: 
        - Respect the order of steps within each clip, and then across clips.
        - Do not hallucinate actions when none are stated.
        - Do not invent steps.
        - Do not change the order of steps.
        """  
    }, 
    "prompt6": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Task of this prompt:
        Analyze short video clips and their audio.
        Extract the lowest-level human actions being performed.
        Each clip will have actions being performed to achieve a step in the procedure (there may be one action to multiple actions).
        From each clip, get all the low-level actions a human is performing based on what is seen in the video and heard from the audio (or read from the transcription of the audio). 
        Examples of low level actions: pick up flask, pour substance A to 10 mL, set flask down, twist cap, etc. 
        Extraction of actions should only come from the video and audio, do not make up any actions to logically fill in gaps. Do not hallucinate low-level actions from the video clip and audio. 
        Use audio transcription and video clip to see low level actions and what materials/substances are being used.
        Notice physical actions performed by the human, chemicals, tools, and containers used, any measurements or quantities involved

        Goal of this task: 
        Extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.
        For example, you should be extracting low level actions like pick up tweezers, drop tweezers, grab vial from shelf, place hand on cap of **substance name**, twist off cap, etc. 
        The actions should be low-level. They should each logical step the human is performing in the video. 
        The goal is to find all low-level actions in the video clip (not broad/high level actions).
        Use video as the primary source of truth.
        Use audio as a guide to deduce information about the procedure in the clip. The human is voicing their procedure, so use this to achieve the goal. 

        The input of these clips are short video clips and their transcriptions (audio). 

        The output of this task should be a valid json file of the actions done in the video. Do not include explanations, markdown, or commentary.

        Guidelines:
                - List actions in the order they occur.
                - Include tools, chemicals, and measurements when visible or stated.
                - Do NOT infer intent or future steps beyond what is visible or stated.
                - Do not hallucinate actions when none are visible or stated.


        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. 
        Task of this prompt: 
        Take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow. 
        Actions are found within each file. Each file represents a clip of the full procedural video, so action order should be preserved from file to file and within the file. 
        For example, a file may have the first action as picking up a paper, drop paper, pick up flask. These actions should stay in order and no new actions should be added. The next file might be place flask on weighing machine, add substance, etc. The previous action from the previous file (pick up flask) should connect to the next file’s first action (place flask on weighing machine). 
        Doing this, it should create an overall task graph of the entire procedure.
        Given all the list of actions (in list form or JSON), make a task graph going from one step to the other. 
        Order should be preserved within a video clip’s actions (so within the file, the order of the actions should stay the same).
        Order from one list of actions to the next should stay the same (do not hallucinate and create actions to fill in gaps from one list of actions to the next). 

        Goal of this task:
        Return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.
        Based on JSON/list of actions given from each video clip (all these clips make up the procedure), create a mermaid graph to represent a logical task graph of the entire procedural flow. 
        Construct a clean task graph showing the entire procedural flow.
        Use standard nodes, like A, B, C... for steps and use arrows to connect one low level action/step to the next, building the entire procedure.

        The input of this task is a JSON file or a list of actions performed in each short video clip. 

        The output of this task should be a mermaid flowchart of the overall task graph that represents the entire procedural workflow. 

        Guidelines: 
        - Respect the order of steps within each clip, and then across clips.
        - Do not hallucinate actions when none are stated.
        - Do not invent steps.
        - Do not change the order of steps.
        """  
    }, 
    "prompt7": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Task of this prompt:
        Analyze short video clips and their audio.
        Extract the lowest-level human actions being performed.
        Each clip will have actions being performed to achieve a step in the procedure (there may be one action to multiple actions).
        From each clip, get all the low-level actions a human is performing based on what is seen in the video and heard from the audio (or read from the transcription of the audio). 
        Examples of low level actions: pick up flask, pour substance A to 10 mL, set flask down, twist cap, etc. 
        Extraction of actions should only come from the video and audio, do not make up any actions to logically fill in gaps. Do not hallucinate low-level actions from the video clip and audio. 
        Use audio transcription and video clip to see low level actions and what materials/substances are being used.

        Goal of this task: 
        Extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.
        For example, you should be extracting low level actions like pick up tweezers, drop tweezers, grab vial from shelf, place hand on cap of **substance name**, twist off cap, etc. 
        The actions should be low-level. They should each logical step the human is performing in the video. 
        The goal is to find all low-level actions in the video clip (not broad/high level actions).
        Use video as the primary source of truth.
        Use audio as a guide to deduce information about the procedure in the clip. The human is voicing their procedure, so use this to achieve the goal. 

        Input given:
                1. A video clip (8–10 seconds) of a human performing actions in a materials science lab.
                2. An optional audio transcription of the clip.
                - If the audio contains spoken instructions, use it to inform the actions.
                - If the audio is background noise, ignore it.

        The output of this task should be a valid json file of the actions done in the video. Do not include explanations, markdown, or commentary.

        Guidelines:
        - List actions in the order they occur.
        - Include tools, chemicals, and measurements when visible or stated.
        - Do NOT infer intent or future steps beyond what is visible or stated.
        - Do not hallucinate actions when none are visible or stated.

        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. 
        Task of this prompt: 
        Take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow. 
        Actions are found within each file. Each file represents a clip of the full procedural video, so action order should be preserved from file to file and within the file. 
        For example, a file may have the first action as picking up a paper, drop paper, pick up flask. These actions should stay in order and no new actions should be added. The next file might be place flask on weighing machine, add substance, etc. The previous action from the previous file (pick up flask) should connect to the next file’s first action (place flask on weighing machine). 
        Doing this, it should create an overall task graph of the entire procedure.
        Given all the list of actions (in list form or JSON), make a task graph going from one step to the other. 
        Order should be preserved within a video clip’s actions (so within the file, the order of the actions should stay the same).
        Order from one list of actions to the next should stay the same (do not hallucinate and create actions to fill in gaps from one list of actions to the next). 

        Goal of this task:
        Return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.
        Based on JSON/list of actions given from each video clip (all these clips make up the procedure), create a mermaid graph to represent a logical task graph of the entire procedural flow. 
        Construct a clean task graph showing the entire procedural flow.
        Use standard nodes, like A, B, C... for steps and use arrows to connect one low level action/step to the next, building the entire procedure.

        Input given: 
        One text file formatted like a JSON file or a list of actions
        You are given an ordered list of atomic human actions split into clips.
        Each action may include a conditional or repeat information.
        This text file is a list of JSON files, each representing a short video clip from a materials science lab.
        The file will contain information about the steps in the video clip (which is documented in that file). It will have information about the actions performed by the human (these actions are low-level).
        Each file corresponds to a video clip of the entire procedural video - together all the files make up the whole procedure BUT each file is a video clip’s steps/actions.

        The output of this task should be a mermaid flowchart of the overall task graph that represents the entire procedural workflow. 

        Guidelines: 
        - Respect the order of steps within each clip, and then across clips.
        - Do not hallucinate actions when none are stated.
        - Do not invent steps.
        - Do not change the order of steps.
        """  
    }, 
    "prompt8": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Task of this prompt:
        Analyze short video clips and their audio.
        Extract the lowest-level human actions being performed.
        Each clip will have actions being performed to achieve a step in the procedure (there may be one action to multiple actions).
        From each clip, get all the low-level actions a human is performing based on what is seen in the video and heard from the audio (or read from the transcription of the audio). 
        Examples of low level actions: pick up flask, pour substance A to 10 mL, set flask down, twist cap, etc. 
        Extraction of actions should only come from the video and audio, do not make up any actions to logically fill in gaps. Do not hallucinate low-level actions from the video clip and audio. 
        Use audio transcription and video clip to see low level actions and what materials/substances are being used.

        Goal of this task: 
        Extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.
        For example, you should be extracting low level actions like pick up tweezers, drop tweezers, grab vial from shelf, place hand on cap of **substance name**, twist off cap, etc. 
        The actions should be low-level. They should each logical step the human is performing in the video. 
        The goal is to find all low-level actions in the video clip (not broad/high level actions).
        Use video as the primary source of truth.
        Use audio as a guide to deduce information about the procedure in the clip. The human is voicing their procedure, so use this to achieve the goal. 

        Input given:
                1. A video clip (8–10 seconds) of a human performing actions in a materials science lab.
                2. An optional audio transcription of the clip.
                - If the audio contains spoken instructions, use it to inform the actions.
                - If the audio is background noise, ignore it.

        Output:
        Return only valid JSON to list all the actions/steps in the video clip.
        Do not include explanations, markdown, or commentary.

                Use this exact format:

                {{
                "clip_index": {clip_index},
                "actions": [
                    {{
                    "step": 1,
                    "action": "pick up reagent bottle"
                    }},
                    {{
                    "step": 2,
                    "action": "scoop powder (≈5 g)",
                    }},
                    {{
                    "step": 3,
                    "action": "shake excess powder off scoop"
                    }}
                ]
                }}    

        Guidelines:
        - List actions in the order they occur.
        - Include tools, chemicals, and measurements when visible or stated.
        - Do NOT infer intent or future steps beyond what is visible or stated.
        - Do not hallucinate actions when none are visible or stated.


        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. 
        Task of this prompt: 
        Take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow. 
        Actions are found within each file. Each file represents a clip of the full procedural video, so action order should be preserved from file to file and within the file. 
        For example, a file may have the first action as picking up a paper, drop paper, pick up flask. These actions should stay in order and no new actions should be added. The next file might be place flask on weighing machine, add substance, etc. The previous action from the previous file (pick up flask) should connect to the next file’s first action (place flask on weighing machine). 
        Doing this, it should create an overall task graph of the entire procedure.
        Given all the list of actions (in list form or JSON), make a task graph going from one step to the other. 
        Order should be preserved within a video clip’s actions (so within the file, the order of the actions should stay the same).
        Order from one list of actions to the next should stay the same (do not hallucinate and create actions to fill in gaps from one list of actions to the next). 

        Goal of this task:
        Return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.
        Based on JSON/list of actions given from each video clip (all these clips make up the procedure), create a mermaid graph to represent a logical task graph of the entire procedural flow. 
        Construct a clean task graph showing the entire procedural flow.
        Use standard nodes, like A, B, C... for steps and use arrows to connect one low level action/step to the next, building the entire procedure.

        Input given: 
        One text file formatted like a JSON file or a list of actions
        You are given an ordered list of atomic human actions split into clips.
        Each action may include a conditional or repeat information.
        This text file is a list of JSON files, each representing a short video clip from a materials science lab.
        The file will contain information about the steps in the video clip (which is documented in that file). It will have information about the actions performed by the human (these actions are low-level).
        Each file corresponds to a video clip of the entire procedural video - together all the files make up the whole procedure BUT each file is a video clip’s steps/actions.

        Output:
                - Return a mermaid flowchart of the overall task graph based on the files (which are made based on the video clips of the entire procedural video. All clips will make up the whole video, so the whole procedure. Make an overall task graph of the entire procedure)
                - Do not include explanations or markdown.xw    
                - Use conditional branching only when necessary, without extra labels.
                - Output the Mermaid diagram using inline node labels (e.g., A[action] --> B[action]), and do not define nodes separately before listing edges.

                Use this exact format:

                graph TD
                    A[Open cabinet door] --> B[Reach into cabinet]
                    B --> C[Retrieve powder bottle]
                    C --> D[Close cabinet door]
                    D --> E[Hold powder bottle]
                    E --> F[Move bottle to tray]
                    F --> G[Place bottle in tray]
                    G --> H[Optional step?]
                    H --> yes --> I[next step]
                    H --> no --> J[pick up spoon]
                    I --> J

                Return a mermaid graph to represent the task graph's order, sequence, loops, and steps. 

        Guidelines: 
        - Respect the order of steps within each clip, and then across clips.
        - Do not hallucinate actions when none are stated.
        - Do not invent steps.
        - Do not change the order of steps.
        """  
    }, 
    "prompt9": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Task of this prompt:
        Analyze short video clips and their audio.
        Extract the lowest-level human actions being performed.
        Each clip will have actions being performed to achieve a step in the procedure (there may be one action to multiple actions).
        From each clip, get all the low-level actions a human is performing based on what is seen in the video and heard from the audio (or read from the transcription of the audio). 
        Examples of low level actions: pick up flask, pour substance A to 10 mL, set flask down, twist cap, etc. 
        Extraction of actions should only come from the video and audio, do not make up any actions to logically fill in gaps. Do not hallucinate low-level actions from the video clip and audio. 
        Use audio transcription and video clip to see low level actions and what materials/substances are being used.

        Goal of this task: 
        Extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.
        For example, you should be extracting low level actions like pick up tweezers, drop tweezers, grab vial from shelf, place hand on cap of **substance name**, twist off cap, etc. 
        The actions should be low-level. They should each logical step the human is performing in the video. 
        The goal is to find all low-level actions in the video clip (not broad/high level actions).
        Use video as the primary source of truth.
        Use audio as a guide to deduce information about the procedure in the clip. The human is voicing their procedure, so use this to achieve the goal. 

        Input given:
                1. A video clip (8–10 seconds) of a human performing actions in a materials science lab.
                2. An optional audio transcription of the clip.
                - If the audio contains spoken instructions, use it to inform the actions.
                - If the audio is background noise, ignore it.

        Output:
        Return only valid JSON to list all the actions/steps in the video clip.
        Do not include explanations, markdown, or commentary.

                Use this exact format:

                {{
                "clip_index": {clip_index},
                "actions": [
                    {{
                    "step": 1,
                    "action": "pick up reagent bottle"
                    }},
                    {{
                    "step": 2,
                    "action": "scoop powder (≈5 g)",
                    }},
                    {{
                    "step": 3,
                    "action": "shake excess powder off scoop"
                    }}
                ]
                }}    

        Guidelines:
        Output only actions that are explicitly visible in the video or explicitly stated in the transcript.
        Actions must be listed in exact temporal order as they occur.
        Each action must be a single observable event (no combining steps).
        Do not infer goals, intent, purpose, cause, or outcome.
        Do not infer missing steps, preparation, or follow-up actions.
        Include an object, tool, chemical, or material only if it is clearly visible or named.
        Include quantities, measurements, labels, or settings only if they are explicitly shown or spoken.
        If an object, substance, or action cannot be clearly identified, label it as “unidentified” or “unclear”.
        Do not use domain knowledge or common-sense reasoning.
        Do not rename, clarify, or normalize objects beyond what is shown or said.
        Do not add safety assumptions or implied procedures.
        Do not predict or describe what happens after the final visible action.
        If no action is visible or stated, output “No observable action.”
        Use neutral, literal language (e.g., “move hand toward container”).
        Do not paraphrase beyond the minimum needed to describe the visible action.


        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. 
        Task of this prompt: 
        Take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow. 
        Actions are found within each file. Each file represents a clip of the full procedural video, so action order should be preserved from file to file and within the file. 
        For example, a file may have the first action as picking up a paper, drop paper, pick up flask. These actions should stay in order and no new actions should be added. The next file might be place flask on weighing machine, add substance, etc. The previous action from the previous file (pick up flask) should connect to the next file’s first action (place flask on weighing machine). 
        Doing this, it should create an overall task graph of the entire procedure.
        Given all the list of actions (in list form or JSON), make a task graph going from one step to the other. 
        Order should be preserved within a video clip’s actions (so within the file, the order of the actions should stay the same).
        Order from one list of actions to the next should stay the same (do not hallucinate and create actions to fill in gaps from one list of actions to the next). 

        Goal of this task:
        Return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.
        Based on JSON/list of actions given from each video clip (all these clips make up the procedure), create a mermaid graph to represent a logical task graph of the entire procedural flow. 
        Construct a clean task graph showing the entire procedural flow.
        Use standard nodes, like A, B, C... for steps and use arrows to connect one low level action/step to the next, building the entire procedure.

        Input given: 
        One text file formatted like a JSON file or a list of actions
        You are given an ordered list of atomic human actions split into clips.
        Each action may include a conditional or repeat information.
        This text file is a list of JSON files, each representing a short video clip from a materials science lab.
        The file will contain information about the steps in the video clip (which is documented in that file). It will have information about the actions performed by the human (these actions are low-level).
        Each file corresponds to a video clip of the entire procedural video - together all the files make up the whole procedure BUT each file is a video clip’s steps/actions.

        Output:
                - Return a mermaid flowchart of the overall task graph based on the files (which are made based on the video clips of the entire procedural video. All clips will make up the whole video, so the whole procedure. Make an overall task graph of the entire procedure)
                - Do not include explanations or markdown.xw    
                - Use conditional branching only when necessary, without extra labels.
                - Output the Mermaid diagram using inline node labels (e.g., A[action] --> B[action]), and do not define nodes separately before listing edges.

                Use this exact format:

                graph TD
                    A[Open cabinet door] --> B[Reach into cabinet]
                    B --> C[Retrieve powder bottle]
                    C --> D[Close cabinet door]
                    D --> E[Hold powder bottle]
                    E --> F[Move bottle to tray]
                    F --> G[Place bottle in tray]
                    G --> H[Optional step?]
                    H --> yes --> I[next step]
                    H --> no --> J[pick up spoon]
                    I --> J

                Return a mermaid graph to represent the task graph's order, sequence, loops, and steps. 

        Guidelines: 
        Maintain the exact chronological order of actions:
        First within a single clip, then in the order the clips appear.
        Output actions only if they are explicitly visible or explicitly stated.
        If an action is not shown or said, do not output it.
        Do not add, assume, infer, or fill in missing actions.
        Do not create intermediate steps.
        Do not reorder, merge, split, or skip actions.
        If two actions happen at the same time, list them in the order they become visible.
        If no action occurs in a clip, output “No observable action.”

        """  
    },
    "prompt10": {
        "video_clip_prompt": 
        """
        You are an expert observer in a material science laboratory
        Task of this prompt:
        Analyze short video clips and their audio.
        Extract the lowest-level human actions being performed.
        Each clip will have actions being performed to achieve a step in the procedure (there may be one action to multiple actions).
        From each clip, get all the low-level actions a human is performing based on what is seen in the video and heard from the audio (or read from the transcription of the audio). 
        Examples of low level actions: pick up flask, pour substance A to 10 mL, set flask down, twist cap, etc. 
        Extraction of actions should only come from the video and audio, do not make up any actions to logically fill in gaps. Do not hallucinate low-level actions from the video clip and audio. 
        Use audio transcription and video clip to see low level actions and what materials/substances are being used.
        Note: Actions in a clip may repeat (e.g., pick up → set down → pick up again). 
        If a human retries, corrects, undoes, or repeats an action, each occurrence must be listed as a separate action in the order it occurs. 
        Do not collapse repeated actions into one step. If the human continues forward after an error, record only the actions that actually occur.

        Goal of this task: 
        Extract the primitive, low-level actions performed by the human and list these actions from the video clips and audio.
        For example, you should be extracting low level actions like pick up tweezers, drop tweezers, grab vial from shelf, place hand on cap of **substance name**, twist off cap, etc. 
        The actions should be low-level. They should each logical step the human is performing in the video. 
        The goal is to find all low-level actions in the video clip (not broad/high level actions).
        Use video as the primary source of truth.
        Use audio as a guide to deduce information about the procedure in the clip. The human is voicing their procedure, so use this to achieve the goal. 
        The goal is to capture exactly what the human does, even if the sequence includes repetition, correction, pauses, or alternative action paths. 
        There is no single correct pathway assumed. 
        If the human deviates, redoes a step, or proceeds differently than expected, record the observed actions exactly as they happen without interpretation or correction.

        Input given:
                1. A video clip (8–10 seconds) of a human performing actions in a materials science lab.
                2. An optional audio transcription of the clip.
                - If the audio contains spoken instructions, use it to inform the actions.
                - If the audio is background noise, ignore it.

        Output:
        Return only valid JSON to list all the actions/steps in the video clip.
        Do not include explanations, markdown, or commentary.
        If an action is repeated, undone, or reattempted, include each instance as its own step with a new step number. 
        Step numbers must strictly increase and reflect the actual order of actions, even if actions appear redundant.

                Use this exact format:

                {{
                "clip_index": {clip_index},
                "actions": [
                    {{
                    "step": 1,
                    "action": "pick up reagent bottle"
                    }},
                    {{
                    "step": 2,
                    "action": "scoop powder (≈5 g)",
                    }},
                    {{
                    "step": 3,
                    "action": "shake excess powder off scoop"
                    }}
                ]
                }}    

        Guidelines:
        Output only actions that are explicitly visible in the video or explicitly stated in the transcript.
        Actions must be listed in exact temporal order as they occur.
        Each action must be a single observable event (no combining steps).
        Do not infer goals, intent, purpose, cause, or outcome.
        Do not infer missing steps, preparation, or follow-up actions.
        Include an object, tool, chemical, or material only if it is clearly visible or named.
        Include quantities, measurements, labels, or settings only if they are explicitly shown or spoken.
        If an object, substance, or action cannot be clearly identified, label it as “unidentified” or “unclear”.
        Do not use domain knowledge or common-sense reasoning.
        Do not rename, clarify, or normalize objects beyond what is shown or said.
        Do not add safety assumptions or implied procedures.
        Do not predict or describe what happens after the final visible action.
        If no action is visible or stated, output “No observable action.”
        Use neutral, literal language (e.g., “move hand toward container”).
        Do not paraphrase beyond the minimum needed to describe the visible action.
        Actions may repeat, and repetition must be recorded explicitly.
        Do not assume a canonical procedure or “correct” workflow.
        If a human pauses, hesitates, redoes, or reverses an action, record each observable action separately.
        If multiple possible pathways exist, record only the pathway that actually occurs in the clip.
        Do not remove, fix, or reinterpret actions that appear mistaken or inefficient.
        Temporal order is absolute, even when actions seem redundant or contradictory.

        Transcription of the video's audio:
        \"\"\"{transcription}\"\"\"
        """, 
        "task_graph_prompt": 
        """
        You are an expert in procedural task modeling. 
        Task of this prompt: 
        Take all the actions (list of actions) from the clips analyzed and make an overall task graph to show the entire procedural flow. 
        Actions are found within each file. Each file represents a clip of the full procedural video, so action order should be preserved from file to file and within the file. 
        For example, a file may have the first action as picking up a paper, drop paper, pick up flask. These actions should stay in order and no new actions should be added. The next file might be place flask on weighing machine, add substance, etc. The previous action from the previous file (pick up flask) should connect to the next file’s first action (place flask on weighing machine). 
        Doing this, it should create an overall task graph of the entire procedure.
        Given all the list of actions (in list form or JSON), make a task graph going from one step to the other. 
        Order should be preserved within a video clip’s actions (so within the file, the order of the actions should stay the same).
        Order from one list of actions to the next should stay the same (do not hallucinate and create actions to fill in gaps from one list of actions to the next). 
        Actions may repeat, be undone, or be re-attempted across or within clips. 
        If an action occurs more than once, each occurrence must be represented as a distinct node in the task graph, preserving the exact order observed. 
        Do not collapse, deduplicate, or normalize repeated actions. 
        If the human deviates from a previous action and then continues forward, represent only the actions that actually occur.

        Goal of this task:
        Return a mermaid graph to represent the task graph’s (made from all of the actions found in the clips) order, sequence, loops, and steps.
        Based on JSON/list of actions given from each video clip (all these clips make up the procedure), create a mermaid graph to represent a logical task graph of the entire procedural flow. 
        Construct a clean task graph showing the entire procedural flow.
        Use standard nodes, like A, B, C... for steps and use arrows to connect one low level action/step to the next, building the entire procedure.
        There is no assumed single correct procedure. The task graph must reflect the actual observed procedural flow, including repetition, retries, corrections, pauses, or alternative paths taken by the human. 
        If the procedure branches due to a mistake, correction, or choice, represent only the branches that are explicitly present in the input actions, without inventing or completing missing paths.

        Input given: 
        One text file formatted like a JSON file or a list of actions
        You are given an ordered list of atomic human actions split into clips.
        Each action may include a conditional or repeat information.
        This text file is a list of JSON files, each representing a short video clip from a materials science lab.
        The file will contain information about the steps in the video clip (which is documented in that file). It will have information about the actions performed by the human (these actions are low-level).
        Each file corresponds to a video clip of the entire procedural video - together all the files make up the whole procedure BUT each file is a video clip’s steps/actions.

        Output:
                - Return a mermaid flowchart of the overall task graph based on the files (which are made based on the video clips of the entire procedural video. All clips will make up the whole video, so the whole procedure. Make an overall task graph of the entire procedure)
                - Do not include explanations or markdown.xw    
                - Use conditional branching only when necessary, without extra labels.
                - Output the Mermaid diagram using inline node labels (e.g., A[action] --> B[action]), and do not define nodes separately before listing edges.
                - Repeated actions must appear as separate nodes, even if the action text is identical.
                - If an action leads back to a previously performed action, represent this as a loop only when the loop is explicitly supported by repeated actions in the input.
                - Conditional branches should be included only if conditional behavior is explicitly present in the input action lists.
                - Do not invent alternative branches, fallback paths, or “ideal” flows.

                Use this exact format:

                graph TD
                    A[Open cabinet door] --> B[Reach into cabinet]
                    B --> C[Retrieve powder bottle]
                    C --> D[Close cabinet door]
                    D --> E[Hold powder bottle]
                    E --> F[Move bottle to tray]
                    F --> G[Place bottle in tray]
                    G --> H[Optional step?]
                    H --> yes --> I[next step]
                    H --> no --> J[pick up spoon]
                    I --> J

                Return a mermaid graph to represent the task graph's order, sequence, loops, and steps. 

        Guidelines: 
        Maintain the exact chronological order of actions:
        First within a single clip, then in the order the clips appear.
        Output actions only if they are explicitly visible or explicitly stated.
        If an action is not shown or said, do not output it.
        Do not add, assume, infer, or fill in missing actions.
        Do not create intermediate steps.
        Do not reorder, merge, split, or skip actions.
        If two actions happen at the same time, list them in the order they become visible.
        If no action occurs in a clip, output “No observable action.”
        Do not assume a canonical, optimal, or correct workflow.
        Do not collapse repeated actions into a single node.
        Do not remove or reinterpret actions that appear mistaken, redundant, or inefficient.
        If the human retries a step, represent the retry as a new node in sequence.
        If multiple pathways are possible in theory, include only the pathway(s) explicitly present in the input actions.
        Use loops and branches only when directly implied by repeated or conditional actions in the input.
        Preserve absolute temporal order across clips, even when actions appear contradictory or repetitive.

        """  
    }
}