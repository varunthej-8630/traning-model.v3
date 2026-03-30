# Extreme Scale & Accuracy Refactor Plan

The user requested support for scaling the application to handle "lakhs" (100,000+) of images for "long term training" to yield highly accurate models. 

Currently, if you feed the browser 100,000 images, the browser’s memory will crash because it tries to store a visual thumbnail of every single image, and it completely freezes the main thread while processing this massive queue.

## User Review Required

> [!CAUTION]
> **Major Memory Paradigm Shift**
> To prevent your browser from crashing with 100,000 images, we must drop the feature that stores a visual "thumbnail" for every single image. Instead, we will only keep the raw math (feature embeddings), saving **gigabytes** of RAM and allowing infinite scaling!
> The UI will still show a few thumbnails just to let you know it's working, but it won't permanently store all 100,000 pictures in memory. 

> [!IMPORTANT]
> **Training for Extreme Accuracy**
> I am going to upgrade the "Training Engine" to support deeper, highly accurate neural networks. 
> I will implement **Early Stopping**. This means you can set `EPOCHS` to `1000` (long-term training), but the system will automatically stop if the accuracy stops getting better. This prevents the model from "overlearning" and ruining its accuracy!

## Proposed Changes

### [MODIFY] trainer.js
- **Memory Optimization (Thumbnails):** 
  - Update `takeSnapshot()`, `processImage()`, and `processVideo()` to completely stop generating and pushing `imageData` into RAM.
  - The `cls.samples` array will be pure math features (Float32).
  - Modify `renderThumbnails()` to use static icons or just show the active count, completely removing heavy canvas DOM elements.
- **Queue Unblocking (Anti-Freeze):** 
  - In `processFilesForClass()`, add a `yield` mechanism (`setTimeout(..., 0)`) every 25 files so the browser doesn't lock up and display the "Page Unresponsive" error when dropping 50,000 files at once.
- **Advanced Neural Network:** 
  - Change the `model.fit()` setup to include `tf.callbacks.earlyStopping({ monitor: 'val_acc', patience: 15 })`.
  - Upgrade the `tf.sequential()` layers to be slightly wider (e.g. `512` units instead of `256`) to handle much more complex data from lakhs of images, giving higher accuracy.
  - Update `cfg-epochs` limit up to `500` or `1000` epochs for serious long-term training.

### [MODIFY] trainer.html
- Change the `EPOCHS` slider `max` from `200` to `1000` to support massive long-term training runs.
- Add a tiny "Pro Mode" memory tag to the dataset manager explaining that heavy datasets are safely handled.

## Verification
- Add 5,000 files all at once to a single class: verify the browser does not freeze, the progress logged perfectly, and RAM usage stays flat.
- Set Epochs to 1000 and run. Verify that if validation accuracy plateaus for 15 epochs, it automatically safely halts and saves the model.
