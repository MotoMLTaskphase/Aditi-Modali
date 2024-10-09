import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('userbase.csv')

print(df.tail())
print(df.isnull().sum())  # to check null values

# Univariate analysis on 'Age'
print(df['Age'].describe())
print(f"Skewness of 'Age': {df['Age'].skew()}")

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
df['Age'].plot(kind='kde', title='Age Distribution (KDE)')
plt.subplot(1,2,2)
df['Age'].plot(kind='box', title='Age Boxplot')
plt.show()

# Univariate analysis on 'Monthly Revenue'
print(df['Monthly Revenue'].describe())

plt.figure(figsize=(6,4))
df['Monthly Revenue'].plot(kind='box', title='Monthly Revenue Boxplot')
plt.show()  # You mentioned there are no outliers, this helps verify

# Univariate analysis on 'Gender'
print(df['Gender'].value_counts())  # Check if both genders are almost equal

# Univariate analysis on 'Device'
print(df['Device'].value_counts())  # Almost equal quantities of devices

# Univariate analysis on 'Subscription Type'
df['Subscription Type'].value_counts().plot(kind='pie', autopct='%0.1f%%', figsize=(6,6))
plt.title('Subscription Type Distribution')
plt.show()

# Univariate analysis on 'Country'
df['Country'].value_counts().plot(kind='pie', autopct='%0.1f%%', figsize=(6,6))
plt.title('Country Distribution')
plt.show()

# Univariate analysis on 'Plan Duration'
print(df['Plan Duration'].value_counts())  # Verify if most took 1-month plans

# Bivariate analysis
plt.figure(figsize=(8,6))
sns.boxplot(x='Age', y='Monthly Revenue', data=df)
plt.title('Age by Monthly Revenue')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Device', y='Subscription Type', data=df)
plt.title('Device by Subscription Type')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x='Subscription Type', y='Monthly Revenue', data=df)
plt.title('Monthly Revenue by Subscription Type')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(pd.crosstab(df['Country'], df['Gender']), annot=True, fmt='d', cmap='coolwarm')
plt.title('Country vs Gender Heatmap')
plt.show()

# Feature Engineering

df['Join Date'] = pd.to_datetime(df['Join Date'], format='%d-%m-%y', errors='coerce')  # Coerce handles invalid dates
df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'], format='%d-%m-%y', errors='coerce')

# join duration and days since the last payment
df['Join Duration'] = (df['Last Payment Date'] - df['Join Date']).dt.days
df['Days Since Last Payment'] = (pd.to_datetime('today') - df['Last Payment Date']).dt.days

# Droping columns
df = df.drop(columns=['Join Date', 'Last Payment Date'])

# Final DataFrame 
print(df.head())


