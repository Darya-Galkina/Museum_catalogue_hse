
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.optim import AdamW 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from transformers import get_linear_schedule_with_warmup
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score, roc_auc_score
import warnings
from collections import Counter
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
from scipy.sparse import csr_matrix
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F


warnings.filterwarnings('ignore')
# Конфигурация
CONFIG = {
    'json_path': '/root/projects/iconclass/data_iconclass_words.json',
    'image_dir': '/root/projects/iconclass/images_iconclass',
    # 'model_name': 'openai/clip-vit-base-patch32',
    'model_name': 'openai/clip-vit-large-patch14',  
    'fusion_method': 'weighted_sum',  # Альтернатива 'concat'
    'batch_size': 16, 
    'max_epochs': 5,  
    'patience': 7,
    'lr': 5e-5,
    'warmup_steps': 1000,
    'weight_decay': 0.01,
    'num_workers': 4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'val_size': 0.15,
    'test_size': 0.15,
    'random_state': 42,
    'text_prefix': "This image contains tags:\n",
    'max_text_length': 77,
    # 'fusion_method': 'concat',
    'save_dir': '/root/projects/iconclass/for_test/saved_model',
    'save_plots': True,
    'gradient_accumulation_steps': 4,
    'mixed_precision': True,
    'max_tags_per_image': 20,
    'min_tag_count': 100,
    'top_k_tags': 5000,    #  количество самых частых тегов
    'pos_weight_smooth': 15.0,  # Сглаживание для весов классов
    'class_weights': 'balanced',  # Автоматическая балансировка
    'sampler': 'multilabel'      # Использовать MultilabelStratifiedSampler
}
def get_transforms(train=True):
    """Возвращает трансформации для обучения или валидации"""
    if train:
        return transforms.Compose([
            RandomResizedCrop(224, scale=(0.8, 1.0)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def load_data(json_path):
    """Загрузка и подготовка данных с фильтрацией тегов"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Сбор уникальных тегов и их частот
    tag_descriptions = {}
    tag_counts = Counter()
    
    for img_name, img_tags in data.items():
        img_tags_set = set()
        for tag_desc in img_tags:
            if 'tag' in tag_desc:
                parts = tag_desc.split('tag')
                desc_part = parts[0].replace("'", "").strip(" ,-")
                tag = parts[-1].strip()
                tag_descriptions[tag] = f"{desc_part} ({tag})"
                img_tags_set.add(tag)
        tag_counts.update(img_tags_set)
    
    
    selected_tags = [tag for tag, count in tag_counts.items() 
                if count >= CONFIG['min_tag_count']]
    
    selected_tags = sorted(selected_tags, key=lambda x: tag_counts[x], reverse=True)[:CONFIG['top_k_tags']]
    
    # Фильтрация тегов по минимальному количеству
    # selected_tags = [tag for tag, count in tag_counts.items() if count >= CONFIG['min_tag_count']]
    
    if not selected_tags:
        raise ValueError("No tags left after filtering!")
    
    # Расчет покрытия
    coverage = sum(tag_counts[tag] for tag in selected_tags) / sum(tag_counts.values())
    print(f"Selected {len(selected_tags)} tags (coverage: {coverage:.2%})")
    print(f"Top tags: {tag_counts.most_common(10)}")
    
    # Создание маппингов
    tag2idx = {tag: idx for idx, tag in enumerate(selected_tags)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    
    # Сбор данных с фильтрацией
    image_paths = []
    text_descriptions = []
    label_rows = []
    skipped = 0
    
    for img_name, img_tags in data.items():
        img_path = os.path.join(CONFIG['image_dir'], img_name)
        if not os.path.exists(img_path):
            skipped += 1
            continue
            
        label_vec = np.zeros(len(selected_tags), dtype=np.float32)
        processed_tags = []
        
        for tag_desc in img_tags:
            if 'tag' not in tag_desc:
                continue
                
            tag = tag_desc.split('tag')[-1].strip()
            if tag in tag2idx:
                label_vec[tag2idx[tag]] = 1.0
                processed_tags.append(tag_descriptions[tag])
        
        if label_vec.sum() > 0:
            image_paths.append(img_path)
            text_desc = CONFIG['text_prefix'] + "\n".join(
                [f"- {tag}" for tag in processed_tags[:CONFIG['max_tags_per_image']]]
            )
            text_descriptions.append(text_desc)
            label_rows.append(label_vec)
        else:
            skipped += 1
    
    # Создаем разреженную матрицу лейблов
    labels = csr_matrix(np.vstack(label_rows)) if label_rows else csr_matrix((0, len(selected_tags)))
    
    # Расчет весов для балансировки классов
    class_counts = np.array(label_rows).sum(axis=0)
    pos_weights = (len(label_rows) - class_counts) / (class_counts + CONFIG['pos_weight_smooth'])
    # pos_weights = torch.clamp(pos_weights, max=10.0)  # Ограничьте максимальный вес

    print(f"\nDataset stats:")
    print(f"Images: {len(image_paths)} (skipped: {skipped})")
    print(f"Tags: {len(selected_tags)}")
    print(f"Label matrix shape: {labels.shape}")
    print(f"Class balance (min/avg/max): {class_counts.min():.0f}/{class_counts.mean():.0f}/{class_counts.max():.0f}")
    
    return image_paths, text_descriptions, labels, tag2idx, idx2tag, torch.FloatTensor(pos_weights)
class MultiLabelDataset(Dataset):
    def __init__(self, image_paths, text_descriptions, labels, processor, transform=None, is_train=False):
        self.image_paths = image_paths
        self.text_descriptions = text_descriptions
        self.labels = csr_matrix(labels) if not isinstance(labels, csr_matrix) else labels
        self.processor = processor
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        for _ in range(3):  # Делаем несколько попыток загрузки
            try:
                image = Image.open(self.image_paths[idx]).convert('RGB')

                
                text = self.text_descriptions[idx]
                
                inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt",
                    padding='max_length',
                    max_length=CONFIG['max_text_length'],
                    truncation=True
                )
                
                label = torch.FloatTensor(self.labels[idx].toarray().squeeze(0))
                
                return {
                    'pixel_values': inputs['pixel_values'].squeeze(),
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'labels': label
                }
            except Exception as e:
                print(f"Error loading {self.image_paths[idx]}: {str(e)}")
                idx = np.random.randint(0, len(self))  # Пробуем другое изображение
        
        # Если все попытки неудачны, возвращаем нули
        return {
            'pixel_values': torch.zeros((3, 224, 224)),
            'input_ids': torch.zeros((CONFIG['max_text_length'],), dtype=torch.long),
            'attention_mask': torch.zeros((CONFIG['max_text_length'],), dtype=torch.long),
            'labels': torch.zeros(self.labels.shape[1], dtype=torch.float)
        }
class CLIPMultiLabelClassifier(nn.Module):
    """Модель классификатора"""
    def __init__(self, num_labels):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(CONFIG['model_name'])
        
        # Замораживаем CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Настройка fusion
        self.fusion_dim = self.clip.config.projection_dim * 2 if CONFIG['fusion_method'] == 'concat' else self.clip.config.projection_dim
        
        # архитектура
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(1024)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_labels)
        )
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.cuda.amp.autocast(enabled=CONFIG['mixed_precision']):
            outputs = self.clip(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            if CONFIG['fusion_method'] == 'concat':
                fused = torch.cat([outputs.image_embeds, outputs.text_embeds], dim=1)
            else:
                fused = outputs.image_embeds + outputs.text_embeds
                
            fused = self.fusion_proj(fused)
            return self.classifier(fused)

def compute_metrics(y_true, y_prob):
    """Расчет метрик с обработкой редких классов"""
    def find_optimal_threshold(y_true, y_prob, thresholds):
        best_f1 = 0
        best_thresh = 0.1  # Дефолтное значение
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh

    thresholds = np.linspace(0.01, 0.3, 20)
    per_class_thresholds = [find_optimal_threshold(y_true[:,i], y_prob[:,i], thresholds) 
                        for i in range(y_true.shape[1])]
    y_pred = np.array([(y_prob[:,i] >= t).astype(int) for i, t in enumerate(per_class_thresholds)]).T

    metrics = {
        'threshold': float(best_threshold),  # Сохраняем оптимальный порог
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_samples': f1_score(y_true, y_pred, average='samples', zero_division=0),
    }
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob, average='macro')
    except:
        metrics['roc_auc'] = 0.0
    
    return metrics
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=1.0, gamma_pos=0.0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    def forward(self, outputs, targets):
        logpt = -F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        pt = torch.exp(logpt)
        # Асимметричная модификация
        pt_neg = (1 - pt).clamp(max=self.clip)
        pt_pos = pt.clamp(max=1 - self.clip)
        
        loss = -((1 - targets) * (pt_neg ** self.gamma_neg) * logpt + 
                 targets * (pt_pos ** self.gamma_pos) * logpt)
        
        return loss.mean()
def train_model():
    """Основная функция обучения"""
    torch.cuda.empty_cache()
    gc.collect()
    
    # Загрузка данных
    image_paths, texts, labels, tag2idx, idx2tag, pos_weights = load_data(CONFIG['json_path'])
    num_labels = len(tag2idx)
    
    # Разделение данных
    X_train, X_test, y_train, y_test, train_texts, test_texts = train_test_split(
        image_paths, labels, texts,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    X_train, X_val, y_train, y_val, train_texts, val_texts = train_test_split(
        X_train, y_train, train_texts,
        test_size=CONFIG['val_size']/(1-CONFIG['test_size']),
        random_state=CONFIG['random_state']
    )
    
    # Инициализация процессора CLIP
    processor = CLIPProcessor.from_pretrained(CONFIG['model_name'])
    
    # Создание датасетов
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    train_dataset = MultiLabelDataset(X_train, train_texts, y_train, processor, train_transform, True)
    val_dataset = MultiLabelDataset(X_val, val_texts, y_val, processor, val_transform)
    test_dataset = MultiLabelDataset(X_test, test_texts, y_test, processor, val_transform)
    
    # Балансировка классов через WeightedRandomSampler
    sample_weights = [1.0 / (np.sum(label) + 1e-6) for label in y_train.toarray()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Создание DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # Инициализация модели
    model = CLIPMultiLabelClassifier(num_labels).to(CONFIG['device'])
    # Функция потерь с весами классов
    criterion = AsymmetricLoss(gamma_neg=1.0, gamma_pos=0.5, clip=0.05).to(CONFIG['device'])

    # Оптимизатор и шедулер
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['max_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Обучение
    best_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG['mixed_precision'])
    
#
    for epoch in range(CONFIG['max_epochs']):
        torch.cuda.empty_cache()
        gc.collect()
    
    # Фаза обучения
        model.train()
        epoch_loss = 0.0
    
        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["max_epochs"]}')):
            inputs = {k: v.to(CONFIG['device']) for k, v in batch.items()}
        
            with torch.cuda.amp.autocast(enabled=CONFIG['mixed_precision']):
                outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
                loss = criterion(outputs, inputs['labels'])
                loss = loss / CONFIG['gradient_accumulation_steps']
        
            scaler.scale(loss).backward()
        
            if (step + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        
            epoch_loss += loss.item() * CONFIG['gradient_accumulation_steps']
    
        history['train_loss'].append(epoch_loss / len(train_loader))
        
        # Фаза валидации
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                inputs = {k: v.to(CONFIG['device']) for k, v in batch.items()}
                outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
                
                val_loss += criterion(outputs, inputs['labels']).item()
                all_probs.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(inputs['labels'].cpu().numpy())
        
        # Расчет метрик
        val_loss /= len(val_loader)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_labels, all_probs)
        
        history['val_loss'].append(val_loss)
        history['val_f1'].append(metrics['f1_macro'])
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"Val ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Сохранение лучшей модели
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            print(f"Saved new best model with F1 Macro: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"Early stopping after {CONFIG['patience']} epochs without improvement")
                break
    
    # Загрузка лучшей модели для тестирования
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'best_model.pth')))
    
    # Тестирование
    test_probs, test_labels = [], []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            inputs = {k: v.to(CONFIG['device']) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})
            test_probs.append(torch.sigmoid(outputs).cpu().numpy())
            test_labels.append(inputs['labels'].cpu().numpy())
    
    test_metrics = compute_metrics(np.concatenate(test_labels), np.concatenate(test_probs))
    
    print("\nFinal Test Metrics:")
    print(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    
    # # Сохранение результатов
    # results = {
    #     'config': CONFIG,
    #     'history': history,
    #     'test_metrics': test_metrics,
    #     'tag2idx': tag2idx,
    #     'idx2tag': idx2tag
    # }
    
    results = {
    'config': {k: v for k, v in CONFIG.items() if not isinstance(v, torch.device)},
    'history': history,
    'test_metrics': test_metrics,
    'tag2idx': tag2idx,
    'idx2tag': idx2tag
    }
    
    with open(os.path.join(CONFIG['save_dir'], 'results.json'), 'w') as f:
        json.dump(results, f)
    
    return model, tag2idx, idx2tag
if __name__ == '__main__':
    model, tag2idx, idx2tag = train_model()
