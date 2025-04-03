import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from tqdm import tqdm
import random
import re
import os
from collections import Counter

# Configuração de seed para reprodutibilidade
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed()

# Configurações
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
MODEL_NAME = 'xlm-roberta-base'  # Modelo multilíngue
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIN_SAMPLES_PER_CLASS = 50  # Mínimo de amostras por classe após aumento de dados

# Carregamento e preparação dos dados
def load_data(file_path):
    """
    Carrega os dados de um arquivo CSV ou XLSX.
    Espera colunas 'texto' (ou 'comentario') e 'classe'
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Formato de arquivo não suportado. Use CSV ou Excel.")
    
    # Verificar e renomear colunas se necessário
    if 'comentario' in df.columns and 'texto' not in df.columns:
        df.rename(columns={'comentario': 'texto'}, inplace=True)
    
    # Verificar se as colunas necessárias existem
    if 'texto' not in df.columns or 'classe' not in df.columns:
        raise ValueError("O arquivo deve conter as colunas 'texto'/'comentario' e 'classe'")
    
    # Remover duplicatas exatas
    print(f"Tamanho original: {len(df)}")
    df.drop_duplicates(subset=['texto'], inplace=True)
    print(f"Tamanho após remoção de duplicatas: {len(df)}")
    
    # Remover linhas com textos vazios
    df = df[df['texto'].notna() & (df['texto'].str.strip() != '')]
    
    # Verificar distribuição das classes
    class_distribution = df['classe'].value_counts()
    print(f"Número de classes: {len(class_distribution)}")
    print(f"Classe com menos amostras: {class_distribution.min()} amostras")
    print(f"Classe com mais amostras: {class_distribution.max()} amostras")
    
    return df

# Técnicas de aumento de dados
class TextAugmentation:
    def __init__(self, lang='por'):
        self.lang = lang
        # Inicializar os aumentadores de texto
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet', lang=self.lang)
        self.back_translation_aug = naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-pt-en',
            to_model_name='Helsinki-NLP/opus-mt-en-pt'
        )
        self.random_word_aug = naw.RandomWordAug(action="swap")
        self.char_aug = nac.RandomCharAug(action="substitute", aug_char_p=0.1)
    
    def _clean_text(self, text):
        """Limpa o texto removendo caracteres especiais e espaços extras"""
        if isinstance(text, str):
            text = re.sub(r'\s+', ' ', text)  # Remove espaços extras
            text = text.strip()
            return text
        return ""
    
    def augment(self, text, techniques=None):
        """
        Aplica técnicas de aumento de dados ao texto
        
        Args:
            text (str): Texto a ser aumentado
            techniques (list): Lista de técnicas a serem aplicadas
                               Opções: 'synonym', 'back_translation', 'random_word', 'char'
                               Se None, escolhe aleatoriamente
        
        Returns:
            str: Texto aumentado
        """
        if not text or len(text.strip()) < 5:
            return text
        
        clean_text = self._clean_text(text)
        if not clean_text:
            return text

        # Se não especificou técnicas, escolhe aleatoriamente
        if not techniques:
            techniques = random.choice([
                ['synonym'],
                ['back_translation'],
                ['random_word'],
                ['char'],
                ['synonym', 'char'],
                ['random_word', 'char']
            ])
        
        augmented_text = clean_text
        
        try:
            for technique in techniques:
                if technique == 'synonym':
                    augmented_text = self.synonym_aug.augment(augmented_text)
                elif technique == 'back_translation':
                    augmented_text = self.back_translation_aug.augment(augmented_text)
                elif technique == 'random_word':
                    augmented_text = self.random_word_aug.augment(augmented_text)
                elif technique == 'char':
                    augmented_text = self.char_aug.augment(augmented_text)
        except Exception as e:
            print(f"Erro ao aumentar texto: {e}")
            return clean_text
            
        return augmented_text

def balance_dataset(df, min_samples=MIN_SAMPLES_PER_CLASS):
    """
    Equilibra o dataset usando técnicas de aumento de dados
    para que cada classe tenha pelo menos min_samples amostras
    """
    augmenter = TextAugmentation()
    class_counts = df['classe'].value_counts()
    
    augmented_data = []
    
    for classe, count in tqdm(class_counts.items(), desc="Equilibrando classes"):
        if count < min_samples:
            # Selecionar todas as amostras da classe
            class_samples = df[df['classe'] == classe]
            samples_needed = min_samples - count
            
            # Se precisar de mais amostras do que o dobro disponível, 
            # use técnicas mais intensivas
            if samples_needed > count:
                # Múltiplas técnicas por amostra
                for _, row in class_samples.iterrows():
                    n_augmentations = max(1, samples_needed // count)
                    for _ in range(n_augmentations):
                        techniques = random.sample(['synonym', 'back_translation', 'random_word', 'char'], 
                                                 k=random.randint(1, 3))
                        augmented_text = augmenter.augment(row['texto'], techniques)
                        augmented_data.append({
                            'texto': augmented_text,
                            'classe': classe,
                            'augmented': True
                        })
                        samples_needed -= 1
                        if samples_needed <= 0:
                            break
                    if samples_needed <= 0:
                        break
            else:
                # Selecionar amostras aleatórias para aumentar
                augmentation_candidates = class_samples.sample(n=samples_needed, replace=True)
                for _, row in augmentation_candidates.iterrows():
                    technique = random.choice([['synonym'], ['back_translation'], ['random_word'], ['char']])
                    augmented_text = augmenter.augment(row['texto'], technique)
                    augmented_data.append({
                        'texto': augmented_text,
                        'classe': classe,
                        'augmented': True
                    })
    
    # Adicionar coluna de controle no dataset original
    df['augmented'] = False
    
    # Combinar dados originais com dados aumentados
    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data)
        balanced_df = pd.concat([df, augmented_df], ignore_index=True)
        print(f"Adicionadas {len(augmented_data)} amostras aumentadas")
        return balanced_df
    else:
        print("Nenhuma amostra aumentada necessária")
        return df

# Dataset e DataLoader
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Função de treinamento
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc="Treinando")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(data_loader)

# Função de avaliação
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Avaliando"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    return predictions, actual_labels

# Função principal
def main(data_path, output_dir='modelo_xlm_roberta'):
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar dados
    df = load_data(data_path)
    
    # Equilibrar dataset
    balanced_df = balance_dataset(df)
    
    # Verificar a distribuição após o balanceamento
    print("\nDistribuição após balanceamento:")
    class_distribution = balanced_df['classe'].value_counts()
    print(f"Classe com menos amostras: {class_distribution.min()} amostras")
    print(f"Classe com mais amostras: {class_distribution.max()} amostras")
    
    # Codificar as classes
    le = LabelEncoder()
    balanced_df['label'] = le.fit_transform(balanced_df['classe'])
    
    # Salvar o codificador para uso posterior
    label_encoder_map = dict(zip(le.classes_, le.transform(le.classes_)))
    with open(os.path.join(output_dir, 'label_encoder.txt'), 'w', encoding='utf-8') as f:
        for classe, indice in label_encoder_map.items():
            f.write(f"{classe}\t{indice}\n")
    
    # Dividir em treino, validação e teste
    train_df, temp_df = train_test_split(
        balanced_df, 
        test_size=0.3, 
        stratify=balanced_df['label'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        stratify=temp_df['label'],
        random_state=42
    )
    
    print(f"Tamanho do conjunto de treino: {len(train_df)}")
    print(f"Tamanho do conjunto de validação: {len(val_df)}")
    print(f"Tamanho do conjunto de teste: {len(test_df)}")
    
    # Inicializar tokenizador e modelo
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(le.classes_)
    )
    model.to(DEVICE)
    
    # Preparar datasets
    train_dataset = CommentDataset(
        texts=train_df['texto'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    val_dataset = CommentDataset(
        texts=val_df['texto'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    test_dataset = CommentDataset(
        texts=test_df['texto'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Preparar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Preparar otimizador e scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Treinar modelo
    best_val_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"\nÉpoca {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        print(f"Perda média de treinamento: {train_loss:.4f}")
        
        # Avaliar no conjunto de validação
        val_predictions, val_labels = evaluate(model, val_loader, DEVICE)
        val_report = classification_report(val_labels, val_predictions, zero_division=0, output_dict=True)
        val_accuracy = val_report['accuracy']
        
        print(f"Acurácia de validação: {val_accuracy:.4f}")
        
        # Salvar o melhor modelo
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(os.path.join(output_dir, 'best_model'))
            tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))
            print(f"Melhor modelo salvo com acurácia de validação: {best_val_accuracy:.4f}")
    
    # Carregar o melhor modelo para avaliação final
    best_model = XLMRobertaForSequenceClassification.from_pretrained(
        os.path.join(output_dir, 'best_model')
    )
    best_model.to(DEVICE)
    
    # Avaliar no conjunto de teste
    test_predictions, test_labels = evaluate(best_model, test_loader, DEVICE)
    test_report = classification_report(test_labels, test_predictions, zero_division=0)
    test_confusion = confusion_matrix(test_labels, test_predictions)
    
    print("\nRelatório de classificação no conjunto de teste:")
    print(test_report)
    
    # Converter índices para nomes de classes
    class_names = {i: classe for classe, i in label_encoder_map.items()}
    pred_class_names = [class_names[pred] for pred in test_predictions]
    true_class_names = [class_names[label] for label in test_labels]
    
    # Salvar previsões e labels
    results_df = pd.DataFrame({
        'texto': test_df['texto'].values,
        'classe_real': true_class_names,
        'classe_prevista': pred_class_names,
        'correto': [p == t for p, t in zip(pred_class_names, true_class_names)]
    })
    
    results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
    
    # Salvar relatório de classificação
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(test_report)
    
    print(f"Avaliação completa. Resultados salvos em {output_dir}")

# Função para fazer previsões com o modelo treinado
def predict(text, model_dir, tokenizer_dir, label_encoder_path):
    # Carregar modelo e tokenizador
    model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Carregar mapeamento do label encoder
    label_map = {}
    with open(label_encoder_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                classe, indice = parts
                label_map[int(indice)] = classe
    
    # Codificar o texto
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Fazer previsão
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    
    # Obter o nome da classe prevista
    predicted_class = label_map[prediction.item()]
    confidence_score = confidence.item()
    
    return {
        'texto': text,
        'classe_prevista': predicted_class,
        'confianca': confidence_score
    }

# Exemplo de uso
if __name__ == "__main__":
    # Parâmetros
    DATA_PATH = "caminho/para/seus_dados.csv"  # Caminho para seu dataset (CSV ou Excel)
    OUTPUT_DIR = "modelo_xlm_roberta"  # Diretório para salvar o modelo
    
    # Treinar modelo
    main(DATA_PATH, OUTPUT_DIR)
    
    # Exemplo de como fazer previsões
    # exemplo_texto = "Este é um exemplo de comentário para classificar"
    # resultado = predict(
    #     exemplo_texto,
    #     os.path.join(OUTPUT_DIR, 'best_model'),
    #     os.path.join(OUTPUT_DIR, 'tokenizer'),
    #     os.path.join(OUTPUT_DIR, 'label_encoder.txt')
    # )
    # print(f"Texto: {resultado['texto']}")
    # print(f"Classe prevista: {resultado['classe_prevista']}")
    # print(f"Confiança: {resultado['confianca']:.4f}")
