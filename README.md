# Vesion

Vesion is an experiment in open-world object detection for Anveshak. The synthetic test bench runs on a CPU and generates its own labelled images of shapes. It exercises the path from random initialization to a checkpoint that can reproduce its predictions.

Inspect the extracted competition dataset with `python -m vesion.round1_data round1 --output runs/round1_audit/summary.json`. The audit checks each manifest against the files on disk and validates the released training annotations. `Round1TrainingTiles` loads Set A or Set B as RGB tensors with variable-length box targets in the original tile coordinates. The test split is checked against its manifest without decoding images.

Run the from-scratch real-image pilot from the project root:

```powershell
python -m vesion.real_pilot --device cpu
```

On Kaggle, attach the competition data as a private input if its terms allow it. Run from the project directory and replace the example path with the actual directory containing `class_list.csv`, `dataset_A`, and `dataset_B`. With two T4s selected, use the preinstalled CUDA versions of PyTorch and torchvision and launch one process per GPU:

```bash
torchrun --standalone --nproc-per-node=2 -m vesion.real_pilot \
  --root /kaggle/input/datasets/your-private-input/round1 \
  --output /kaggle/working/vesion_pilot --device cuda --stage A \
  --base-tiles 40 --epochs-a 12 --threads 2
```

This example samples up to 40 crops per active base class. Every epoch prints its training time and saves `stage_A_last.pt` with weights, optimizer state, and mixed-precision state. To continue in another session, keep the same split options and batch size, then add `--resume /path/to/stage_A_last.pt` to the command. The epoch count is the total budget, including completed epochs.

When ready for Set B, replace `--stage A` with `--stage B` and set `--epochs-b`. It reads `stage_A.pt` from the output directory by default. If you mounted the checkpoint from a saved notebook version, use `--stage-a /path/to/stage_A.pt`. Keep the same crop and split options. Compare `raw_box_recall` in `metrics.json` to see whether better boxes reach the class decision.

The pilot uses separate source tiles for training, calibration, and validation. It reserves every Set B tile ID before sampling Set A because the release reuses those IDs across the two sets. It keeps the native pixel scale in 256 × 256 crops and hides three Set A class IDs during training to simulate unknowns. Stage B reloads the Stage A weights, uses Set A replay, and skips background loss on Set B crops because those annotations are partial. It compares confidence rejection, prototype rejection, and prototype rejection with extra boxes on the same validation crops. The pilot outputs `stage_A.pt`, `stage_B.pt`, prediction CSVs, and `metrics.json` under the chosen output directory. The scores are for class-centred crops and proxy unknowns. An official run requires training on the released class list and predicting on full tiles with the official submission pipeline.

```powershell
python -m pip install -r requirements.txt
python -m vesion.synthetic
python -m unittest discover -s tests
```

The experiment trains on circles and squares in Stage A. Stage B loads that checkpoint, activates a class slot already in the detector, and learns triangles from 12 labelled images mixed with earlier training examples. Triangles and diamonds are unknown in the Stage A test. Diamonds remain unknown in Stage B. The same test images let us compare old-class accuracy before and after the update.

The detector predicts objectness and class scores separately. Above the objectness threshold, it assigns `unknown` when the top class probability is below the classification threshold. It tunes both thresholds on a separate calibration split with pentagons and plus signs as stand-ins for unknown objects. The test shapes never set the thresholds. The known-only baseline uses identical candidate boxes and objectness filtering, then gives each box its best known label.

`runs/synthetic/metrics.json` reports both scoring modes, old-class mAP before and after the update, new-class AP, and unknown proposal recall. Proposal recall counts unknown objects with a candidate box at IoU ≥ 0.5 before the class decision. Compare it with final unknown recall to see whether to improve the boxes or the labelling rule first. Synthetic objects are deliberately simple, and training scenes have complete annotations. The results exercise the pipeline and leave performance on partially labelled satellite imagery to a separate experiment.

Each stage writes a checkpoint, prediction CSV, and matching ground truth CSV. The run reloads each checkpoint and checks that its predictions reproduce the original CSV byte for byte. To reproduce predictions and score them independently from the saved files:

```powershell
python -m vesion.synthetic --checkpoint runs/synthetic/stage_b.pt --output runs/reproduced
python -m vesion.evaluate runs/reproduced/predictions.csv runs/synthetic/stage_b_truth.csv --known-classes circle square triangle
```

The local evaluator uses the published Anveshak brief. It implements all-point AP for known classes, unknown precision/recall/F1, the known-versus-unknown overlap rule, and OE-Score. Its interpretation keeps higher-ranked surviving rows in overlapping chains and matches each prediction to the highest-overlap available same-label ground truth. Pass `--known-classes` to include labels absent from the ground truth. The header for ground truth is `image_id,class_name,xmin,ymin,xmax,ymax`. Use the organisers’ validator for official submission checks, including test image IDs and tile bounds.

Compare prototype rejection with softmax after the Stage B update:

```powershell
python -m vesion.compare_prototypes runs/synthetic/stage_b.pt
python -m vesion.synthetic --checkpoint runs/prototype_comparison/prototype_balanced.pt --output runs/prototype_reproduced
```

The comparison uses the same detector weights, candidate boxes, and objectness cutoff for every method. It averages features from correctly localized training objects to form one prototype per class and estimates shared within-class feature variation. A box gets the `unknown` label when its closest prototype is too far away. Accepted boxes use the original classifier. The balanced setting caps the known mAP drop at two percentage points on the calibration split. That cap does not apply to test results, and the Stage B calibration split has no labelled triangles.

The report in `runs/prototype_comparison/metrics.json` includes calibration scores and scores on 600 fresh synthetic tiles. Both prototype settings write a checkpoint that includes the feature statistics and threshold, and the experiment checks that reloading each checkpoint reproduces its prediction CSV.

Measure how unlabelled objects affect the detector's ability to propose boxes:

```powershell
python -m vesion.test_unlabelled_objects
```

Both runs start from the same random weights and use the same labelled circles and squares. The second run adds one unlabelled diamond to 30% of the training tiles. Both are evaluated on the same 600 fresh tiles. `runs/unlabelled_objects/metrics.json` records class-agnostic box hits at IoU ≥ 0.5 for several objectness cutoffs and per-tile proposal budgets. Triangles serve as a control because neither run sees them during training. This experiment tests proposal recall before any softmax or prototype rejection rule is applied.

Test whether the prototype rule can recover unknowns from boxes below the normal objectness cutoff:

```powershell
python -m vesion.compare_extra_boxes
python -m vesion.compare_extra_boxes --seed 17 --output runs/extra_boxes_repeat
```

Each run compares the same detector trained with unlabelled diamonds. The second prediction CSV preserves each baseline row and adds only `unknown` predictions from the five highest-scoring boxes below the cutoff on each tile. The experiment calibrates the thresholds on a separate set with pentagons and plus signs, then scores both CSVs on 600 fresh tiles. The metrics report records which method the calibration set selected, keeping the baseline in the event of a tie. Pass `--checkpoint` with that run's `detector.pt` to repeat scoring without retraining.
