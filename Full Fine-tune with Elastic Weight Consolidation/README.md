
# Adaptation Strategies in Large Language Models: RQ1 Focus

This repository explores adaptation strategies in large language models (LLMs), with a specific focus on the balance between plasticity (the ability to learn new tasks) and stability (the ability to retain performance on previously learned tasks). This project is aimed at understanding how LLMs can be adapted to new tasks while minimizing catastrophic forgetting. The focal point here is on Research Question 1 (RQ1) and its associated hypotheses, involving a two-step experimental procedure: full fine-tuning on Task A and subsequent adaptation to Task B using Elastic Weight Consolidation (EWC).

## Research Question 1

**RQ1**: To what extent do Full fine-tuning and Elastic Weight Consolidation (EWC) demonstrate a better balance between plasticity and stability in large language models (LLMs), as assessed by task performance retention and catastrophic forgetting mitigation, when adapting to new tasks?

## Hypotheses for RQ1

- **H1a**: Full fine-tuning and Elastic Weight Consolidation (EWC) demonstrate a better balance between plasticity and stability in LLMs when adapting to new tasks, evidenced by significantly higher task performance retention and/or more effective catastrophic forgetting mitigation compared with other methods.

## Experimental Workflow

1. **Task A (Full Fine-Tuning)**: Initially, the model is fully fine-tuned on Task A, laying the groundwork for adaptation capabilities without catastrophic forgetting.
2. **Task B (EWC Fine-Tuning)**: Following Task A, the model is adapted to Task B using Elastic Weight Consolidation (EWC), aiming to mitigate the loss of Task A's performance.

## Directory Structure

- **Scripts**:
  - `data_prep.py`: Prepares and processes data for Task A and Task B.
  - `full_finetune_task_B.py`: Implements full model fine-tuning for Task A.
  - `ewc/`:
    - `ewc_class.py`: Defines the EWC technique, including classes or functions for calculating the Fisher information matrix and applying the EWC penalty.
    - `ewc_finetune_task_B.py`: Applies EWC for fine-tuning on Task B, preserving Task A's knowledge.

## Installation

Ensure Python is installed, then install required dependencies:

```
pip install -r scripts/requirements.txt
```

## Running the Experiments

For full fine-tuning on Task A:

```
python scripts/full_finetune_task_B.py
```

For EWC fine-tuning on Task B:

```
python scripts/ewc/ewc_finetune_task_B.py
```

## Contributing

Contributions to enhance the analysis or improve the implementation are welcome. Please fork this repository and submit a pull request with your changes.

## License

Ivo Tanku Tapang

## Contact

For further questions or to discuss the results, please contact [Ivo Tanku Tapang](mailto:itankutapang@gmail.com).
