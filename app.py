{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "bd9c112a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy import stats"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d25ef368",
   "metadata": {},
   "source": [
    "Price Elasticity Definition\n",
    "\n",
    "This function takes four arguments:\n",
    "\n",
    "**q1: the quantity demanded at price p1\n",
    "**q2: the quantity demanded at price p2\n",
    "**p1: the initial price\n",
    "**p2: the new price\n",
    "\n",
    "It calculates the price elasticity of demand using the formula:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3907783c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def price_elasticity(q1, q2, p1, p2):\n",
    "    return (q2 - q1) / ((q1 + q2) / 2) / (p2 - p1) * ((p1 + p2) / 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6b074e54",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"Price by Data GGN.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "39f6ed95",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>order_date</th>\n",
       "      <th>Price</th>\n",
       "      <th>Demand</th>\n",
       "      <th>Category</th>\n",
       "      <th>Subcategory</th>\n",
       "      <th>Product Description</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>01/10/21</td>\n",
       "      <td>10.0</td>\n",
       "      <td>10.0</td>\n",
       "      <td>Cleaning Essentials</td>\n",
       "      <td>Cleaners</td>\n",
       "      <td>Vim Bar</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>01/10/21</td>\n",
       "      <td>10.0</td>\n",
       "      <td>40.0</td>\n",
       "      <td>Cleaning Essentials</td>\n",
       "      <td>Cleaners</td>\n",
       "      <td>Vim Bar</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>01/10/21</td>\n",
       "      <td>10.0</td>\n",
       "      <td>30.0</td>\n",
       "      <td>Cleaning Essentials</td>\n",
       "      <td>Cleaners</td>\n",
       "      <td>Vim Bar</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>01/10/21</td>\n",
       "      <td>50.0</td>\n",
       "      <td>50.0</td>\n",
       "      <td>Dairy Goodness</td>\n",
       "      <td>Butter &amp; Cheese</td>\n",
       "      <td>Amul Butter</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>01/10/21</td>\n",
       "      <td>50.0</td>\n",
       "      <td>100.0</td>\n",
       "      <td>Dairy Goodness</td>\n",
       "      <td>Butter &amp; Cheese</td>\n",
       "      <td>Amul Butter</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  order_date  Price  Demand             Category      Subcategory  \\\n",
       "0   01/10/21   10.0    10.0  Cleaning Essentials         Cleaners   \n",
       "1   01/10/21   10.0    40.0  Cleaning Essentials         Cleaners   \n",
       "2   01/10/21   10.0    30.0  Cleaning Essentials         Cleaners   \n",
       "3   01/10/21   50.0    50.0       Dairy Goodness  Butter & Cheese   \n",
       "4   01/10/21   50.0   100.0       Dairy Goodness  Butter & Cheese   \n",
       "\n",
       "  Product Description  \n",
       "0             Vim Bar  \n",
       "1             Vim Bar  \n",
       "2             Vim Bar  \n",
       "3         Amul Butter  \n",
       "4         Amul Butter  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "19f377df",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-03-02 10:17:12.951 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/lib/python3.9/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "# Set the title and logo\n",
    "st.set_page_config(page_title='DSDA Price Elasticity Analytics', page_icon=':money_with_wings:')\n",
    "\n",
    "# Define the app layout\n",
    "st.write(\"\"\"\n",
    "# DSDA Price Elasticity Analytics\n",
    "\"\"\")\n",
    "\n",
    "# Upload the CSV file\n",
    "st.set_option('deprecation.showfileUploaderEncoding', False)\n",
    "data_file = st.file_uploader(\"Upload CSV\", type=[\"csv\"])\n",
    "if data_file is not None:\n",
    "    data = pd.read_csv(data_file)\n",
    "\n",
    "    # Define the input variables\n",
    "    st.sidebar.header('Select the input variables:')\n",
    "    x_variable = st.sidebar.selectbox('Select the x variable:', data.columns)\n",
    "    y_variable = st.sidebar.selectbox('Select the y variable:', data.columns)\n",
    "\n",
    "    # Define the regression model\n",
    "    model = LinearRegression()\n",
    "    model.fit(data[[x_variable]], data[y_variable])\n",
    "    elasticity = - model.coef_[0] * np.mean(data[x_variable]) / np.mean(data[y_variable])\n",
    "\n",
    "    # Show the results\n",
    "    st.write('Price elasticity:', elasticity)\n",
    "\n",
    "    # Create a scatter plot\n",
    "    fig, ax = plt.subplots()\n",
    "    ax.scatter(data[x_variable], data[y_variable])\n",
    "    ax.set_xlabel(x_variable)\n",
    "    ax.set_ylabel(y_variable)\n",
    "    ax.set_title('Price Elasticity Analysis')\n",
    "    ax.text(0.05, 0.95, f'Elasticity: {elasticity:.2f}', transform=ax.transAxes, fontsize=14,\n",
    "            verticalalignment='top')\n",
    "    st.pyplot(fig)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1344bfd6",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (1135476472.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  File \u001b[0;32m\"/var/folders/tv/yx4vymss7130s8q6fvyd2kx40000gp/T/ipykernel_939/1135476472.py\"\u001b[0;36m, line \u001b[0;32m1\u001b[0m\n\u001b[0;31m    streamlit run\u001b[0m\n\u001b[0m              ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3babf294",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
