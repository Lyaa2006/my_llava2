# Experiment 3: Catastrophic Description Drift After Task Training

## Goal

Use the checkpoints recorded in `logs/train_UCIT_full_20260703_211225.log` and, after finishing `task n-1` training, directly generate `task n` descriptions.
This report keeps only `2` samples per transition to show the failure pattern compactly.

## Checkpoint Chain From The Training Log

- `Task1_llava_lora` -> generate `Task2 / ArxivQA` descriptions
- `Task2_llava_lora` -> generate `Task3 / VizWiz` descriptions
- `Task3_llava_lora` -> generate `Task4 / IconQA` descriptions
- `Task4_llava_lora` -> generate `Task5 / CLEVR` descriptions
- `Task5_llava_lora` -> generate `Task6 / Flickr30k` descriptions

## Bad Descriptions

### Task1_to_Task2_ArxivQA

- checkpoint: `/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task1_llava_lora`
- dataset: `/mnt/lyaa/my_llava/UCIT/ArxivQA/train_4w.json`

#### Sample 1

- dataset index: `21222`
- image: `ArxivQA/images/1911.02852_1.jpg`
- failure mode: `hallucinated / off-task description`
- source prompt: What can be inferred about the relationship between the true outage line and the identified outage line for lines greater than 20? A. There is a high level of agreement for all lines greater than 20. B. There is no agreement for any line greater than 20. C. There is a moderate level of agreement, but only for certain lines. D. The agreement decreases as the line number increases. - Rationale:The heatmap shows varying shades of color for lines greater than 20, indicating that there is moderate agreement for certain lines but not for others. The color is not uniformly dark or light, suggesting variability in the agreement. Answer with the option's letter from the given choices directly.

**Bad description:**

The image depicts a line of red objects, possibly cherries, arranged in a row. The cherries are spaced out along the line, with some closer to the front and others further back. The line of cherries extends from the left side of the image to the right, creating a visually appealing and organized presentation.

#### Sample 2

- dataset index: `9886`
- image: `ArxivQA/images/1508.00293_0.jpg`
- failure mode: `hallucinated / off-task description`
- source prompt: What can be inferred about the relationship between γ1 and γ2 based on the solid black lines in the figure? A. γ1 and γ2 are directly proportional to each other. B. γ1 increases as γ2 decreases, indicating an inverse relationship. C. γ1 and γ2 are independent of each other. D. The values of γ1 determine discrete steps of γ2. Rationale:The solid black lines are vertical, indicating that for specific ranges of γ1, there are discrete, constant values of γ2. This suggests that as γ1 changes within a certain range, γ2 remains at a fixed level until a threshold is reached, at which point it jumps to another fixed level. Answer with the option's letter from the given choices directly.

**Bad description:**

The image features a three-dimensional representation of a spider web, with the spider web's "spokes" extending outward from a central point. The web is composed of multiple "spokes" that are interconnected, creating a complex and intricate structure. The spider web is displayed in a black and white color scheme, which emphasizes the web's intricate details. The image provides a clear view of the spider web's structure, showcasing its unique and fascinating design.

### Task2_to_Task3_VizWiz

- checkpoint: `/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task2_llava_lora`
- dataset: `/mnt/lyaa/my_llava/UCIT/VizWiz/train.json`

#### Sample 1

- dataset index: `25875`
- image: `VizWiz/train/VizWiz_train_00015790.jpg`
- failure mode: `hallucinated / off-task description`
- source prompt: What is happening in the image? Generate a brief caption for the image.

**Bad description:**

The image features a blind person using a white cane to navigate through a city street. The person is walking down the sidewalk, and the cane is clearly visible in their hand. The street appears to be busy, with multiple cars parked or driving along the road. The scene captures the daily life of a visually impaired individual, showcasing their ability to navigate urban environments with the help of a white cane.

#### Sample 2

- dataset index: `3164`
- image: `VizWiz/train/VizWiz_train_00007930.jpg`
- failure mode: `hallucinated / off-task description`
- source prompt: What is happening in the image? Generate a brief caption for the image.

**Bad description:**

The image features a pink Pepto Bismol tablet, which is a popular antacid medication. The tablet is prominently displayed in the center of the image, with its distinctive pink color and shape easily recognizable. The tablet is placed on a surface, possibly a table or countertop, which is not visible in the image. The focus of the image is on the Pepto Bismol tablet, highlighting its role as a common medication for treating stomach upset and acid reflux.

### Task3_to_Task4_IconQA

- checkpoint: `/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task3_llava_lora`
- dataset: `/mnt/lyaa/my_llava/UCIT/IconQA/train.json`

#### Sample 1

- dataset index: `2373`
- image: `IconQA/iconqa_data/iconqa/train/fill_in_blank/93589/image.png`
- failure mode: `hallucinated / off-task description`
- source prompt: How many triangles are there? Answer the question using a single word or phrase.

**Bad description:**

A row of orange arrows pointing in the same direction

#### Sample 2

- dataset index: `26911`
- image: `IconQA/iconqa_data/iconqa/train/fill_in_blank/21114/image.png`
- failure mode: `hallucinated / off-task description`
- source prompt: How many shapes are orange? Answer the question using a single word or phrase.

**Bad description:**

A computer screen with a circle graph showing the difference between two sets of data

### Task4_to_Task5_CLEVR

- checkpoint: `/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task4_llava_lora`
- dataset: `/mnt/lyaa/my_llava/UCIT/CLEVR/train_4w.json`

#### Sample 1

- dataset index: `35119`
- image: `CLEVR/images/train/CLEVR_train_021226.png`
- failure mode: `answer-mode collapse`
- source prompt: Subtract 0 cyan cylinders. How many objects are left? Answer the question using a single word or phrase.

**Bad description:**

3

#### Sample 2

- dataset index: `6168`
- image: `CLEVR/images/train/CLEVR_train_063601.png`
- failure mode: `answer-mode collapse`
- source prompt: Subtract all spheres. How many objects are left? Answer the question using a single word or phrase.

**Bad description:**

4

### Task5_to_Task6_Flickr30k

- checkpoint: `/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task5_llava_lora`
- dataset: `/mnt/lyaa/my_llava/UCIT/Flickr30k/train_brief_4w.json`

#### Sample 1

- dataset index: `23965`
- image: `Flickr30k/train/2250713748.jpg`
- failure mode: `answer-mode collapse`
- source prompt: What is happening in the image? Generate a brief caption for the image.

**Bad description:**

10

#### Sample 2

- dataset index: `38193`
- image: `Flickr30k/train/1278818225.jpg`
- failure mode: `answer-mode collapse`
- source prompt: What is happening in the image? Generate a brief caption for the image.

**Bad description:**

8

## Conclusion

Even with only two samples per transition, the checkpoint from `task n-1` already fails to produce stable visual descriptions for `task n`.
The later transitions are especially catastrophic, including direct answer collapse (for example `3`) and placeholder-like outputs (for example `[object Object]`).
