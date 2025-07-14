import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model, Parameters

st.set_page_config(
    page_title="Gaussian Fitter",
    page_icon=":abacus:",
    initial_sidebar_state="expanded"
)

def file2data(file):
    data = pd.read_table(file, sep = '\t', decimal = ',', usecols = [0, 1])
    x = np.array(data.iloc[:, 0])
    y = np.array(data.iloc[:, 1])
    return x, y

def gauss(x, a, b, c):
    return a * np.exp( -(x - b)**2 / (2*c**2) )

class SpectrumData:
    def __init__(
        self, 
        x, 
        y,
    ):
        self.x = x
        self.y = y
        self.result_params = None

    def display_range(self, min, max):    
        x = self.x    
        self.x_display_min = min
        self.x_display_min = max
        self.display_mask = (x > min) & (x < max)

    def calc_range(self, min, max):
        x = self.x
        self.x_calc_min = min
        self.x_calc_max = max
        self.calc_mask = (x > min) & (x < max)

    def draw_figure(self):

        d_mask = self.display_mask
        c_mask = self.calc_mask
        x = self.x

        fig, ax = plt.subplots()

        # main plot
        plt.scatter(x[d_mask], y[d_mask], label='data', s=0.5, c='blue')
        # vertical lines
        plt.axvline(x=self.x_calc_min, color='gray', linewidth=1, linestyle='--')
        plt.axvline(x=self.x_calc_max, color='gray', linewidth=1, linestyle='--')  
        # components and sum of init values
        sum_of_gauss = 0
        for i in range(nGauss):
            a = self.a_vals[i]
            b = self.b_vals[i]
            c = self.c_vals[i]
            this_gauss = gauss(x[c_mask], a, b, c)
            sum_of_gauss += this_gauss
            ax.plot(self.x[c_mask], this_gauss, '--', linewidth=1, label=f'Init - gauss{i}')
        ax.plot(self.x[c_mask], sum_of_gauss, '--', linewidth=2, label='Init - sum')
        # components and sum of fit values
        if self.result_params != None:
            sum_of_gauss = 0
            for i in range(nGauss):
                a = self.result_params[f'a{i}']
                b = self.result_params[f'b{i}']
                c = self.result_params[f'c{i}']
                this_gauss = gauss(x[c_mask], a, b, c)
                sum_of_gauss += this_gauss
                ax.plot(self.x[c_mask], this_gauss, linewidth=1, label=f'Fit - gauss{i}')
            ax.plot(self.x[c_mask], sum_of_gauss, linewidth=2, c='red', label='Fit - sum')
        # cosmetics
        ax.grid(False)
        ax.legend()

        return fig 
    
    def set_parameters(self, nGauss):
        
        self.a_vals = [0] * nGauss
        self.b_vals = [0] * nGauss
        self.c_vals = [0] * nGauss
        self.nGauss = nGauss

    def fit(self):
        x = self.x[self.calc_mask]
        y = self.y[self.calc_mask]
        vars = [self.a_vals, self.b_vals, self.c_vals]

        def fit_model(x, **params):
            sum = np.zeros_like(x)
            n = len(params) // 3

            for i in range(n):
                a = params[f'a{i}']
                b = params[f'b{i}']
                c = params[f'c{i}']
                sum += gauss(x, a, b, c)

            return sum

        model = Model(fit_model)
        params = Parameters()

        for i in range(self.nGauss):
            params.add(f'a{i}', value=vars[0][i], min=0)
            params.add(f'b{i}', value=vars[1][i], min=0)
            params.add(f'c{i}', value=vars[2][i], min=0)

        result = model.fit(y, x=x, params=params)
        self.result_params = result.params

file = st.file_uploader("Choose tsv-format file containing your spectrum data")
if file:

    x, y = file2data(file)
    spectrum = SpectrumData(x, y)

    with st.sidebar:

        x_display_min = st.number_input("Min display range value", value=500)
        x_display_max = st.number_input("Max display range value", value=600)
        x_calc_min = st.number_input("Min calculation range value", value=500)
        x_calc_max = st.number_input("Max calculation range value", value=600)
        nGauss = st.number_input("Number of gaussian functions", value=1)

        calculate_fit = st.button('Perform fit')

    spectrum.display_range(x_display_min, x_display_max)
    spectrum.calc_range(x_calc_min, x_calc_max)
    spectrum.set_parameters(nGauss)

    figure_placeholder = st.empty()

    for i in range(nGauss):
        col1, col2, col3 = st.columns([1, 1, 1], gap = "small")
        with col1:
            spectrum.a_vals[i] = st.number_input(f'a{i}', value = 0.5)
        with col2:
            spectrum.b_vals[i] = st.number_input(f'b{i}', value = 550.0)
        with col3:
            spectrum.c_vals[i] = st.number_input(f'c{i}', value = 2.0)

    if calculate_fit:
        spectrum.fit()
        
        rows = []
        for i in range(spectrum.nGauss):
            vars = {}
            for var in ['a', 'b', 'c']:
                vars[var] = spectrum.result_params[f'{var}{i}'].value
            rows.append({'a': vars['a'], 'b': vars['b'], 'c': vars['c']})
        df = pd.DataFrame(rows)
        df.index.name = "No function\parameters"

        st.write("Fitted parameters:")
        st.dataframe(df.style.format("{:.4f}"))

    figure_placeholder.pyplot(fig = spectrum.draw_figure())



    



    

    