使用 'pixel_level_contrastive_learning' 實踐像素級別對比損失。

請依照需求自行設計 dataloader  
image.shape(batch_size, 3(RGB), W, H)
masks.shape(batch_size, 2(class: target, background), W, H)
