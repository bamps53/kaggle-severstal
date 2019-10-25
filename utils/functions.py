import torch


def predict_batch(model, batch_images, tta=False, task='seg'):
    batch_preds = torch.sigmoid(model(batch_images))

    if tta:
        # h_flip
        h_images = torch.flip(batch_images, dims=[3])
        h_batch_preds = torch.sigmoid(model(h_images))
        if task == 'seg':
            batch_preds += torch.flip(h_batch_preds, dims=[3])
        else:
            batch_preds += h_batch_preds

        # v_flip
        v_images = torch.flip(batch_images, dims=[2])
        v_batch_preds = torch.sigmoid(model(v_images))
        if task == 'seg':
            batch_preds += torch.flip(v_batch_preds, dims=[2])
        else:
            batch_preds += v_batch_preds

        # hv_flip
        hv_images = torch.flip(torch.flip(batch_images, dims=[3]), dims=[2])
        hv_batch_preds = torch.sigmoid(model(hv_images))
        if task == 'seg':
            batch_preds += torch.flip(torch.flip(hv_batch_preds, dims=[3]), dims=[2])
        else:
            batch_preds += hv_batch_preds

        batch_preds /= 4

    return batch_preds.detach().cpu().numpy()


def predict_batch_with_softmax(model, batch_images, tta=False, task='seg'):
    batch_preds = torch.softmax(model(batch_images), 1)

    if tta:
        # h_flip
        h_images = torch.flip(batch_images, dims=[3])
        h_batch_preds = torch.softmax(model(h_images), 1)
        if task == 'seg':
            batch_preds += torch.flip(h_batch_preds, dims=[3])
        else:
            batch_preds += h_batch_preds

        # v_flip
        v_images = torch.flip(batch_images, dims=[2])
        v_batch_preds = torch.softmax(model(v_images), 1)
        if task == 'seg':
            batch_preds += torch.flip(v_batch_preds, dims=[2])
        else:
            batch_preds += v_batch_preds

        # hv_flip
        hv_images = torch.flip(torch.flip(batch_images, dims=[3]), dims=[2])
        hv_batch_preds = torch.softmax(model(hv_images), 1)
        if task == 'seg':
            batch_preds += torch.flip(torch.flip(hv_batch_preds, dims=[3]), dims=[2])
        else:
            batch_preds += hv_batch_preds

        batch_preds /= 4

    return batch_preds.detach().cpu().numpy()
