import pandas as pd

# Загрузка тестовых данных и меток
test_data = pd.read_csv("test.csv")
test_labels = pd.read_csv("test_labels.csv")

# Объединение по идентификатору
merged_data = pd.merge(test_data, test_labels, on="id")

# Отфильтровать строки с -1 значениями
merged_data = merged_data[~(merged_data.iloc[:, 2:] == -1).any(axis=1)]

# Очистка индексов
merged_data.reset_index(drop=True, inplace=True)

# Сохранение в новый файл
merged_data.to_csv("cleaned_test_data.csv", index=False)
