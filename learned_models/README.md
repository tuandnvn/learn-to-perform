## Models that has been saved in this directory

To use the models here, you can use method from *test_all_searcher.py*:

```
test_all_searcher.get_default_models (action_types)
```

where action types belong to:

'SlideAround', 'SlideAway', 'SlideNext', 'SlidePast', 'SlideToward'

Notice that there are three models for SlideAround:

SlideAround.mod 1: Update from SlideAround.mod

SlideAround.mod 2: Update from SlideAround.mod 

The code to update is in *incorporate_feedback.ipynb*:

Using demonstrations (experiments/human_evaluation_2d/SlideAround) and grades (experiments/human_evaluation_2d/slidearound.txt)

SlideAround.mod 1: Using index 0 to 14

SlideAround.mod 2: Using index 0 to 14

SlideAround.mod.updated with all indices

SlideAround.mod.updated.2 with all indices but with automatic oracle

SlideAround.mod.updated.updated follows the hot feedback update method