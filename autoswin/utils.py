import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from collections import defaultdict
import random
import torch
from tqdm import tqdm
from torchvision.transforms import functional as TF
from sklearn.manifold import TSNE
from sklearn import metrics as sk_metrics


def normalise_image(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if (max_val - min_val) > 0:
        o_image_normalized = (image - min_val) / (max_val - min_val)

    o_image_scaled = (o_image_normalized * 255.0)

    return o_image_scaled


def mean_channels_if_rgb(image):
    if image.ndim == 3:
        if image.shape[0] == 3:
            return np.mean(image, axis=0)
        else:
            return np.mean(image, axis=0)
    return image


def plot_average_images_per_label(dataloader, label_descriptions, model, device):
    image_dict = {}
    count_dict = {}
    bottom_row_image_dict = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            for i in range(images.size(0)):
                label = labels[i]
                if hasattr(model, "swin"):
                    img = model.swin_pass(images)
                else:
                    img = images[i].cpu().numpy()

                if label in image_dict:
                    image_dict[label] = img
                    count_dict[label] = 1
                else:
                    image_dict[label] = np.zeros((224, 224))
                    image_dict[label] += img
                    count_dict[label] = 1

                original_image_tensor_cpu = images[0].cpu()
                resized_bottom_img_tensor = TF.resize(original_image_tensor_cpu, [224, 224],
                                                      interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
                bottom_img_np = resized_bottom_img_tensor.numpy().squeeze()
                bottom_row_image_dict[label] = bottom_img_np

    n_labels = len(label_descriptions)

    fig, axes = plt.subplots(2, n_labels, figsize=(5 * n_labels, 8.5), squeeze=False)

    if n_labels > 0:
        axes[0, 0].set_ylabel("Swin/Processed", fontsize=9)
        axes[1, 0].set_ylabel("Original (Lanczos)", fontsize=9)

    for label_idx in range(n_labels):
        axes[0, label_idx].set_title(f"Label: {label_descriptions[label_idx]}", fontsize=10)

        if label_idx in count_dict and count_dict[label_idx] > 0:
            avg_img_top = image_dict[label_idx] / count_dict[label_idx]
            avg_img_top_norm = normalise_image(avg_img_top)
            axes[0, label_idx].imshow(avg_img_top_norm, cmap='gray')
        else:
            axes[0, label_idx].imshow(np.zeros((224, 224)), cmap='gray')
        axes[0, label_idx].axis('off')

        if label_idx in bottom_row_image_dict:
            img_bottom = bottom_row_image_dict[label_idx]
            img_bottom_norm = normalise_image(img_bottom)
            axes[1, label_idx].imshow(img_bottom_norm, cmap='gray')
        else:
            axes[1, label_idx].imshow(np.zeros((224, 224)), cmap='gray')
        axes[1, label_idx].axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def plot_random_images(
    dataloader, model, device, num_images=4, seed=42
):
    random.seed(seed)
    selected_indices = sorted(random.sample(range(len(dataloader.dataset)), num_images))

    selected_images = []

    for idx in selected_indices:
        image, label = dataloader.dataset[idx]
        selected_images.append(image)

    fig, axes = plt.subplots(
        2, num_images, figsize=(5 * num_images, 8.5), squeeze=False
    )

    if num_images > 0:
        axes[0, 0].set_ylabel("Processed", fontsize=9)
        axes[1, 0].set_ylabel("Original", fontsize=9)

    for idx in tqdm(range(num_images)):
        with torch.no_grad():
            image = selected_images[idx].unsqueeze(0).to(device)

            if hasattr(model, "swin_with_adapter"):
                image = model.swin_pass(image)
                if isinstance(image, torch.Tensor):
                    image = image.squeeze().detach().cpu().numpy()
            elif hasattr(model, "swin"):
                image = model(image)
                if isinstance(image, torch.Tensor):
                    image = image.squeeze().detach().cpu().numpy()
            else:
                image = model(image)
                if isinstance(image, torch.Tensor):
                    image = image.squeeze().detach().cpu().numpy()

        original_image_tensor_cpu = selected_images[idx].cpu()
        # Match the processed image resolution for fair comparison
        target_size = image.shape[-2:] if image.ndim == 3 else [224, 224]
        resized_bottom_img_tensor = TF.resize(
            original_image_tensor_cpu,
            list(target_size),
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        bottom_img_np = resized_bottom_img_tensor.numpy().squeeze()

        image = mean_channels_if_rgb(image)
        bottom_img_np = mean_channels_if_rgb(bottom_img_np)

        # Normalise for display
        processed_img_norm = normalise_image(image)
        bottom_img_norm = normalise_image(bottom_img_np)

        # Plot
        axes[0, idx].imshow(processed_img_norm, cmap="gray")
        axes[0, idx].set_title(f"Image {idx+1}", fontsize=10)
        axes[0, idx].axis("off")

        axes[1, idx].imshow(bottom_img_norm, cmap="gray")
        axes[1, idx].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def to_ordinal_labels(labels, num_classes):
    N = labels.shape[0]
    ordinals = torch.zeros((N, num_classes - 1), dtype=torch.float32, device=labels.device)
    for i in range(1, num_classes):
        ordinals[:, i - 1] = (labels >= i).float()
    return ordinals


def ordinal_preds_to_class(preds, threshold=0.5):
    if preds.max() > 1 or preds.min() < 0:
        preds = torch.sigmoid(preds)
    return (preds > threshold).sum(dim=1)


def create_tsne_plot(image_features_list, text_features, labels):
    """
    Create t-SNE visualization of image and text embeddings.
    """
    num_classes = len(text_features)

    # Concatenate all image features
    if len(image_features_list) > 0:
        image_features = np.vstack(image_features_list)
    else:
        return plt.figure()  # Return empty figure if no features

    # Combine image and text features
    all_features = np.vstack([image_features, text_features])

    # Create labels for visualization
    image_labels = np.array(labels)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
    embeddings_2d = tsne.fit_transform(all_features)

    # Split back into image and text embeddings
    image_embeddings = embeddings_2d[:len(image_features)]
    text_embeddings = embeddings_2d[len(image_features):]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for classes (using matplotlib's default color cycle)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_classes)))

    # Plot image embeddings by class
    for class_idx in range(num_classes):
        mask = image_labels == class_idx
        if np.any(mask):
            ax.scatter(
                image_embeddings[mask, 0],
                image_embeddings[mask, 1],
                c=[colors[class_idx]],
                alpha=0.6,
                s=20,
                label=f'Images Class {class_idx}'
            )

    # Plot text embeddings with different markers
    for i, (x, y) in enumerate(text_embeddings):
        ax.scatter(
            x, y,
            c=[colors[i]],
            marker='s',
            s=100,
            edgecolors='black',
            linewidth=2,
            label=f'Text Class {i}' if i == 0 else ""  # Only label first one to avoid clutter
        )

        # Add text labels for text embeddings
        ax.annotate(
            f'T{i}',
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black')
        )

    ax.set_title('t-SNE: Image & Text Embeddings', fontsize=14)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')

    # Create custom legend
    legend_elements = []
    for i in range(num_classes):
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=colors[i], markersize=8,
                                      label=f'Class {i} Images', alpha=0.6))
    legend_elements.append(Line2D([0], [0], marker='s', color='w',
                                  markerfacecolor='gray', markersize=10,
                                  markeredgecolor='black', markeredgewidth=2,
                                  label='Text Embeddings'))

    ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def fm_save_correct_profusion_indices(model, dataloader, device, tokenizer, label_descriptions):
    model.eval()

    correct_by_profusion = {}
    incorrect_by_profusion = {}
    correct = 0
    incorrect = 0

    with torch.no_grad():
        text_features = model.encode_text(tokenizer([l for l in label_descriptions], context_length=256).to(device), normalize=True)

        for i, (batch_images, batch_labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            logits = model(batch_images, text_features)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.22).float()

            bplabel = batch_labels[0][:3]
            if torch.sum(bplabel) > 0:
                bplabel = torch.argmax(bplabel).cpu().numpy() + 1
            else:
                bplabel = 0

            pplabel = preds[0][:3]
            if torch.sum(pplabel) > 0:
                pplabel = torch.argmax(pplabel).cpu().numpy() + 1
            else:
                pplabel = 0

            if pplabel == bplabel:
                correct += 1
                if f"{bplabel}" in correct_by_profusion.keys():
                    correct_by_profusion[f"{bplabel}"].append(i)
                else:
                    correct_by_profusion[f"{bplabel}"] = [i]
            else:
                incorrect += 1
                if f"{bplabel}" in incorrect_by_profusion.keys():
                    incorrect_by_profusion[f"{bplabel}"].append(i)
                else:
                    incorrect_by_profusion[f"{bplabel}"] = [i]

    print(correct)
    print(incorrect)
    print(correct_by_profusion)
    print(incorrect_by_profusion)


def plot_combined_conf_mat(confusion_matrix):
    """Plot confusion matrix (binary, multiclass, or multilabel)."""
    from autoswin.metrics import get_cm_for_class

    if confusion_matrix.shape == (2, 2):
        # Binary case
        fig = plt.figure(figsize=(12, 8))
        cm_display = sk_metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[0, 1])
        cm_display.plot(ax=plt.gca())
        plt.title("Confusion Matrix")
    elif len(confusion_matrix.shape) == 3:
        # Multilabel case
        avg_cm = np.mean(confusion_matrix, axis=0)
        norm_cm = avg_cm / (np.sum(avg_cm, axis=1, keepdims=True))
        fig = plt.figure(figsize=(12, 8))
        cm_display = sk_metrics.ConfusionMatrixDisplay(norm_cm, display_labels=[0, 1])
        cm_display.plot(ax=plt.gca())
        plt.title("Normalised Multilabel Confusion Matrix")
    else:
        # Multiclass case
        TN, FP, FN, TP = get_cm_for_class(confusion_matrix, 0)
        bin_conf_matrix = np.array([[TP, FN],
                                    [FP, TN]])
        display_labels = [i for i in range(len(confusion_matrix[0]))]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        cm_display1 = sk_metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=display_labels)
        cm_display1.plot(ax=axes[0], colorbar=False)
        axes[0].set_title("Confusion Matrix")
        cm_display2 = sk_metrics.ConfusionMatrixDisplay(bin_conf_matrix, display_labels=[0, 1])
        cm_display2.plot(ax=axes[1], colorbar=False)
        axes[1].set_title("Binary Confusion Matrix")
        plt.tight_layout()
    return fig
