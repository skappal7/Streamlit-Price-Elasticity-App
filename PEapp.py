{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f128cac9",
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
   "execution_count": null,
   "id": "5a955313",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4aec39d8",
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
