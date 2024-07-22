import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
def numeric_to_time_period(numeric_value):
    year = numeric_value // 12
    month = numeric_value % 12
    if month == 0:
        year -= 1
        month = 12
    return f"{year}-{month:02d}"

#Ucitavanje

df = pd.read_csv('C:/Users/Djordje/Desktop/analiza/estat_ttr00016_en.csv')
df = df[df['geo']!='EU27_2020']
df.drop_duplicates(inplace=True)

# Prikaz informacija o DataFrame-u
#print(df.info())

# DESKRIPTIVNA STATISTIKA

descriptive_stats = df['OBS_VALUE'].describe()

min_time_period = df['TIME_PERIOD'].min()
max_time_period = df['TIME_PERIOD'].max()

max_value_index = df['OBS_VALUE'].idxmax()
max_value = df.loc[max_value_index, 'OBS_VALUE']
max_value_time_period = df.loc[max_value_index, 'TIME_PERIOD'];
max_value_country = df.loc[max_value_index, 'geo']


output_text = (
    f"Analiza deskriptivne statistike za kolonu 'OBS_VALUE':\n"
    f"Podaci su prikazani za period od {min_time_period} do {max_time_period}.\n\n"
    f"Broj vrednosti (count): {descriptive_stats['count']}\n"
    f"Srednja vrednost (mean): {descriptive_stats['mean']:.2f}\n"
    f"Standardna devijacija (std): {descriptive_stats['std']:.2f}\n"
    f"Minimalna vrednost (min): {descriptive_stats['min']:.2f}\n"
    f"Prvi kvartil (25%): {descriptive_stats['25%']:.2f}\n"
    f"Medijana (50%): {descriptive_stats['50%']:.2f}\n"
    f"Treci kvartil (75%): {descriptive_stats['75%']:.2f}\n"
    f"Maksimalna vrednost (max): {descriptive_stats['max']:.2f}\n\n"
    f"Najvise putnika ({max_value}) se vozilo u {max_value_time_period} u zemlji {max_value_country}."
)

print(output_text)

#ANALIZA KORELACIJE

passangers_per_time_period = df.groupby('TIME_PERIOD')['OBS_VALUE'].sum()
print(passangers_per_time_period)

#LINEARNA REGRESIJA

df['YEAR'] = df['TIME_PERIOD'].str.split('-', expand=True)[0].astype(int)
df['MONTH'] = df['TIME_PERIOD'].str.split('-', expand=True)[1].astype(int)
df['TIME_NUMERIC'] = df['YEAR'] * 12 + df['MONTH']

# Priprema podataka za linearnu regresiju
X = df['TIME_NUMERIC'].values.reshape(-1, 1)
y = df['OBS_VALUE'].values

# Kreiranje i fitovanje modela linearnom regresijom
model = LinearRegression()
model.fit(X, y)

# Generisanje buducih vremenskih perioda za predvidjana
future_time_periods = np.arange(df['TIME_NUMERIC'].max() + 1, df['TIME_NUMERIC'].max() + 13, 1)
future_time_periods_numeric = future_time_periods.reshape(-1, 1)


# Predvidjanje buducih vrednosti broja putnika
future_predictions = model.predict(future_time_periods_numeric)

# Ispis predvidjenih buducih vrednosti
for time_period_numeric, prediction in zip(future_time_periods, future_predictions):
    time_period = numeric_to_time_period(time_period_numeric)
    print(f"Budući vremenski period: {time_period}, Predviđen broj putnika: {prediction.astype(int)}")


# 1. Grafikon broja putnika po vremenskim periodima
plt.figure(figsize=(10, 6))
plt.plot(df['TIME_PERIOD'], df['OBS_VALUE'], marker='o', linestyle='-')
plt.xlabel('Vremenski period')
plt.ylabel('Broj putnika')
plt.title('Broj putnika po vremenskim periodima')
plt.xticks(rotation=45)
# Dodavanje FuncFormatter za y-osu
def y_formatter(x, pos):
    return f'{int(x):,}'

plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))
# plt.show()

# 2. Grafikon koji prikazuje trend rasta broja putnika
plt.figure(figsize=(10, 6))
plt.plot(df['TIME_PERIOD'], df['OBS_VALUE'], marker='o', linestyle='-', label='Stvarni podaci')
plt.plot([numeric_to_time_period(tp) for tp in future_time_periods], future_predictions, marker='x', linestyle='--', color='red', label='Predikcije')
plt.xlabel('Vremenski period')
plt.ylabel('Broj putnika')
plt.title('Trend rasta broja putnika')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))
# plt.show()

# Priprema podataka
grouped_by_country = df.groupby('geo')['OBS_VALUE'].sum().reset_index()


fig = go.Figure(data=[go.Table(
    header=dict(values=['Zemlja', 'Ukupan broj putnika'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[grouped_by_country['geo'], grouped_by_country['OBS_VALUE']],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(title='Broj putnika po zemljama')
fig.show()

# Grupisanje podataka po vrednostima polja tra_cov i sumiranje broja putnika za svaku vrednost
grouped_tra_cov = df.groupby('tra_cov')['OBS_VALUE'].sum()

# Crtanje pie grafikona
plt.figure(figsize=(8, 8))
plt.pie(grouped_tra_cov, labels=grouped_tra_cov.index, autopct='%1.1f%%', startangle=140)
plt.title('Udeo trajanja pokrivenosti u ukupnom broju putnika')
plt.axis('equal')  # Čini da se pie grafikon izgleda kao krug
plt.show()