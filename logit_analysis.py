"""
Лабораторна робота №6. Завдання 1. Варіант 9.
Побудова logit-моделі з дискретними залежними змінними.

Автор: Python реалізація
Дата: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Налаштування для кращого відображення
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

print("="*80)
print("ЛАБОРАТОРНА РОБОТА №6. ЗАВДАННЯ 1. ВАРІАНТ 9")
print("Побудова logit-моделі з дискретними залежними змінними")
print("="*80)
print()

# ============================================================================
# КРОК 1: ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
# ============================================================================
print("КРОК 1: Завантаження та підготовка даних")
print("-"*80)

# Завантаження даних
df = pd.read_csv('firms_data.csv')
print("Вихідні дані (всі підприємства):")
print(df)
print()

# Визначення груп за варіантом 9
# Група A (0): підприємства № 1, 3, 4, 5, 6, 8, 24, 29
# Група B (1): підприємства № 14, 15, 16, 21, 23, 27, 28, 41
group_A = [1, 3, 4, 5, 6, 8, 24, 29]
group_B = [14, 15, 16, 21, 23, 27, 28, 41]

# Додавання бінарної змінної group
df['group'] = df['firm_id'].apply(lambda x: 1 if x in group_B else (0 if x in group_A else np.nan))

# Розділення на навчальну вибірку та підприємства для прогнозу
train_df = df[df['group'].notna()].copy()
predict_df = df[df['group'].isna()].copy()

print(f"Навчальна вибірка: {len(train_df)} підприємств")
print(f"  - Група A (0): {len(train_df[train_df['group']==0])} підприємств")
print(f"  - Група B (1): {len(train_df[train_df['group']==1])} підприємств")
print(f"Підприємства для прогнозу: {len(predict_df)} підприємств (№{list(predict_df['firm_id'])})")
print()

print("Навчальна вибірка:")
print(train_df[['firm_id', 'X1', 'X2', 'X3', 'X4', 'X5', 'group']])
print()

# Описова статистика
print("Описова статистика факторних ознак (навчальна вибірка):")
print(train_df[['X1', 'X2', 'X3', 'X4', 'X5']].describe())
print()

# ============================================================================
# КРОК 2: ПОБУДОВА LOGIT-МОДЕЛІ
# ============================================================================
print("="*80)
print("КРОК 2: Побудова logit-моделі")
print("-"*80)

# Підготовка змінних
y_train = train_df['group']
X_train = train_df[['X1', 'X2', 'X3', 'X4', 'X5']]
X_train_const = sm.add_constant(X_train)

# Побудова logit-моделі
logit_model = Logit(y_train, X_train_const)
logit_result = logit_model.fit(method='bfgs', maxiter=100)

# Виведення повного звіту
print("\nПовний звіт logit-моделі:")
print(logit_result.summary())
print()

# Виведення рівняння моделі
print("Рівняння logit-моделі:")
print("P(Y=1|X) = exp(u) / (1 + exp(u))")
print("де u = β₀ + β₁·X₁ + β₂·X₂ + β₃·X₃ + β₄·X₄ + β₅·X₅")
print()
print("Оцінені коефіцієнти:")
for param, value in logit_result.params.items():
    print(f"  {param}: {value:.6f}")
print()

# Формування рівняння з числами
params = logit_result.params
equation = f"u = {params['const']:.4f}"
for i, var in enumerate(['X1', 'X2', 'X3', 'X4', 'X5'], 1):
    sign = '+' if params[var] >= 0 else ''
    equation += f" {sign} {params[var]:.4f}·{var}"
print(f"Конкретне рівняння: {equation}")
print()

# ============================================================================
# КРОК 3: КЛАСИФІКАЦІЯ ТА МАТРИЦЯ ПОМИЛОК (П.1 УМОВИ)
# ============================================================================
print("="*80)
print("КРОК 3: Оцінка кількості правильних класифікацій (п.1)")
print("-"*80)

# Прогнозовані ймовірності
y_pred_proba = logit_result.predict(X_train_const)
train_df['predicted_prob'] = y_pred_proba

# Класифікація при порозі 0.5
y_pred = (y_pred_proba >= 0.5).astype(int)
train_df['predicted_group'] = y_pred

# Таблиця спостережуваних, прогнозованих та залишків
print("Таблиця: Спостережувані, прогнозовані ймовірності та класифікація:")
result_table = train_df[['firm_id', 'group', 'predicted_prob', 'predicted_group']].copy()
result_table.columns = ['Підприємство', 'Реальна група', 'P(B)', 'Прогноз групи']
print(result_table.to_string(index=False))
print()

# Матриця класифікації
cm = confusion_matrix(y_train, y_pred)
print("Матриця класифікації:")
print(f"{'':15} {'Прогноз A (0)':>15} {'Прогноз B (1)':>15}")
print(f"{'Реальна A (0)':15} {cm[0,0]:>15} {cm[0,1]:>15}")
print(f"{'Реальна B (1)':15} {cm[1,0]:>15} {cm[1,1]:>15}")
print()

# Кількість правильних класифікацій
correct_A = cm[0,0]
correct_B = cm[1,1]
total_correct = correct_A + correct_B
total = len(y_train)
percent_correct = (total_correct / total) * 100

print(f"Кількість правильно класифікованих підприємств групи A: {correct_A} з {cm[0,0] + cm[0,1]}")
print(f"Кількість правильно класифікованих підприємств групи B: {correct_B} з {cm[1,0] + cm[1,1]}")
print(f"Загальна кількість правильних класифікацій: {total_correct} з {total}")
print(f"Відсоток правильних класифікацій: {percent_correct:.2f}%")
print()

# Детальний звіт класифікації
print("Детальний звіт класифікації:")
print(classification_report(y_train, y_pred, target_names=['Група A', 'Група B']))
print()

# ============================================================================
# КРОК 4: СЕРЕДНІ ФАКТОРНІ ОЗНАКИ
# ============================================================================
print("="*80)
print("КРОК 4: Обчислення середніх факторних ознак")
print("-"*80)

mean_X = X_train.mean()
print("Середні значення факторних ознак (навчальна вибірка):")
for var, val in mean_X.items():
    print(f"  {var}: {val:.4f}")
print()

# ============================================================================
# КРОК 5: МАРЖИНАЛЬНІ ЕФЕКТИ (П.2 УМОВИ)
# ============================================================================
print("="*80)
print("КРОК 5: Розрахунок середніх маржинальних ефектів (п.2)")
print("-"*80)

# Маржинальні ефекти при середніх значеннях
margeff = logit_result.get_margeff(at='mean', method='dydx')
print("Маржинальні ефекти при середніх значеннях факторів:")
print(margeff.summary())
print()

print("Інтерпретація маржинальних ефектів:")
print("Маржинальний ефект показує, як зміна відповідного фактора на 1 одиницю")
print("впливає на ймовірність належності до групи B (при фіксованих інших факторах).")
print()

margeff_values = margeff.margeff
for i, var in enumerate(['X1', 'X2', 'X3', 'X4', 'X5']):
    effect = margeff_values[i]
    print(f"  {var}: {effect:.6f} - зміна {var} на 1 одиницю {'збільшує' if effect > 0 else 'зменшує'} "
          f"ймовірність належності до групи B на {abs(effect)*100:.4f}%")
print()

# ============================================================================
# КРОК 6: ПРОГНОЗ ДЛЯ ПІДПРИЄМСТВ, ЩО НЕ УВІЙШЛИ У ВИБІРКИ (П.3 УМОВИ)
# ============================================================================
print("="*80)
print("КРОК 6: Прогноз для підприємств, що не увійшли у навчальні вибірки (п.3)")
print("-"*80)

if len(predict_df) > 0:
    X_predict = predict_df[['X1', 'X2', 'X3', 'X4', 'X5']]
    X_predict_const = sm.add_constant(X_predict)
    
    # Прогнозовані ймовірності
    y_predict_proba = logit_result.predict(X_predict_const)
    predict_df['predicted_prob'] = y_predict_proba
    
    # Класифікація
    y_predict_group = (y_predict_proba >= 0.5).astype(int)
    predict_df['predicted_group'] = y_predict_group
    predict_df['predicted_group_name'] = predict_df['predicted_group'].map({0: 'A', 1: 'B'})
    
    print("Прогнозовані ймовірності та класифікація для підприємств, що залишилися:")
    predict_table = predict_df[['firm_id', 'X1', 'X2', 'X3', 'X4', 'X5', 'predicted_prob', 'predicted_group_name']].copy()
    predict_table.columns = ['№', 'X1', 'X2', 'X3', 'X4', 'X5', 'P(B)', 'Прогноз']
    print(predict_table.to_string(index=False))
    print()
    
    print("Інтерпретація:")
    for idx, row in predict_df.iterrows():
        firm_id = int(row['firm_id'])
        prob = row['predicted_prob']
        group = row['predicted_group_name']
        print(f"  Підприємство №{firm_id}: ймовірність належності до групи B = {prob:.4f} ({prob*100:.2f}%)")
        print(f"    → Прогнозована група: {group}")
else:
    print("Немає підприємств для прогнозування.")
print()

# ============================================================================
# КРОК 7: ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
# ============================================================================
print("="*80)
print("КРОК 7: Візуалізація результатів")
print("-"*80)

# Графік 1: Маржинальні ефекти
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 Бар-чарт маржинальних ефектів
ax1 = axes[0, 0]
margeff_df = pd.DataFrame({
    'Фактор': ['X1', 'X2', 'X3', 'X4', 'X5'],
    'Ефект': margeff_values
})
colors = ['green' if x > 0 else 'red' for x in margeff_values]
ax1.barh(margeff_df['Фактор'], margeff_df['Ефект'], color=colors, alpha=0.7)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax1.set_xlabel('Маржинальний ефект')
ax1.set_title('Маржинальні ефекти факторів')
ax1.grid(axis='x', alpha=0.3)

# 1.2 Розподіл прогнозованих ймовірностей
ax2 = axes[0, 1]
train_df_A = train_df[train_df['group'] == 0]
train_df_B = train_df[train_df['group'] == 1]
ax2.hist(train_df_A['predicted_prob'], bins=8, alpha=0.6, label='Група A (реальна)', color='blue', edgecolor='black')
ax2.hist(train_df_B['predicted_prob'], bins=8, alpha=0.6, label='Група B (реальна)', color='orange', edgecolor='black')
ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Поріг класифікації')
ax2.set_xlabel('Прогнозована ймовірність P(B)')
ax2.set_ylabel('Частота')
ax2.set_title('Розподіл прогнозованих ймовірностей')
ax2.legend()
ax2.grid(alpha=0.3)

# 1.3 Матриця класифікації (heatmap)
ax3 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3,
            xticklabels=['Прогноз A', 'Прогноз B'],
            yticklabels=['Реальна A', 'Реальна B'])
ax3.set_title('Матриця класифікації')
ax3.set_ylabel('Реальна група')
ax3.set_xlabel('Прогнозована група')

# 1.4 Scatter plot: реальні групи vs прогнозовані ймовірності
ax4 = axes[1, 1]
for group_val, group_name, color in [(0, 'Група A', 'blue'), (1, 'Група B', 'orange')]:
    subset = train_df[train_df['group'] == group_val]
    ax4.scatter(subset.index, subset['predicted_prob'], 
                label=group_name, alpha=0.7, s=100, color=color, edgecolors='black')
ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Поріг 0.5')
ax4.set_xlabel('Індекс спостереження')
ax4.set_ylabel('Прогнозована ймовірність P(B)')
ax4.set_title('Реальні групи vs Прогнозовані ймовірності')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('logit_analysis_results.png', dpi=300, bbox_inches='tight')
print("Графіки збережено у файл: logit_analysis_results.png")
plt.show()

# Графік 2: Залежність ймовірності від окремих факторів
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
axes2 = axes2.flatten()

for i, var in enumerate(['X1', 'X2', 'X3', 'X4', 'X5']):
    ax = axes2[i]
    
    # Створення діапазону значень для фактора
    x_range = np.linspace(X_train[var].min(), X_train[var].max(), 100)
    
    # Фіксуємо інші фактори на середніх значеннях
    X_plot = pd.DataFrame({v: [mean_X[v]] * 100 for v in ['X1', 'X2', 'X3', 'X4', 'X5']})
    X_plot[var] = x_range
    X_plot_const = sm.add_constant(X_plot, has_constant='add')
    
    # Прогнозовані ймовірності
    y_plot = logit_result.predict(X_plot_const)
    
    # Графік
    ax.plot(x_range, y_plot, linewidth=2, color='purple')
    ax.scatter(X_train[var][train_df['group']==0], [0]*sum(train_df['group']==0), 
               alpha=0.6, s=80, color='blue', label='Група A', marker='o')
    ax.scatter(X_train[var][train_df['group']==1], [1]*sum(train_df['group']==1), 
               alpha=0.6, s=80, color='orange', label='Група B', marker='s')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(var)
    ax.set_ylabel('P(B)')
    ax.set_title(f'Залежність P(B) від {var}')
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend()

# Видалити зайву підграфік
fig2.delaxes(axes2[5])

plt.tight_layout()
plt.savefig('logit_factor_dependencies.png', dpi=300, bbox_inches='tight')
print("Графіки залежностей збережено у файл: logit_factor_dependencies.png")
plt.show()

# ============================================================================
# ВИСНОВКИ
# ============================================================================
print("="*80)
print("ВИСНОВКИ")
print("-"*80)
print()
print("1. Побудовано logit-модель залежності групи підприємства від п'яти факторів:")
print("   X1 (продуктивність праці), X2 (питома вага робітників),")
print("   X3 (коефіцієнт змінності), X4 (питома вага втрат), X5 (фондовіддача).")
print()
print(f"2. Модель показала {percent_correct:.2f}% правильних класифікацій на навчальній вибірці.")
print()
print("3. Маржинальні ефекти показують вплив кожного фактора на ймовірність")
print("   належності до групи B при середніх значеннях інших факторів.")
print()
print("4. Для підприємств, що не увійшли у навчальні вибірки, виконано прогноз")
print("   ймовірності належності до групи B та класифікацію.")
print()
print("5. Графіки демонструють якість моделі, розподіл прогнозів та залежності")
print("   ймовірності від окремих факторів.")
print()
print("="*80)
print("АНАЛІЗ ЗАВЕРШЕНО")
print("="*80)
