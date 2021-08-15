#############################################
# Equations that changed the world
# Based on the classic Ian Stewart book 2012
# Web app to explore the equations
# v0.2 Daniel Lewis 2021
#############################################

import streamlit as st
import altair as alt
import math
import numpy as np
import pandas as pd
import scipy.stats as sstats
import base64
import random
import plotly.graph_objects as go
from fractions import Fraction
import emoji

st.set_page_config(
     page_title='Equations explorer',
     layout="centered",
     initial_sidebar_state="expanded",
)

def write_eq(eqno):

    if eqno == 1:
        st.markdown('''
        __1. Pythagoras' Theorem__
        \n<small>_Pythagoras, 530 BC_</small>
        ''', unsafe_allow_html=True)

        pt_eqtn = ''' a^2 + b^2 = c^2 '''

        st.latex(pt_eqtn)

    elif eqno == 2:
        st.markdown('''
        __2. Logarithms__
        \n<small>_John Napier, 1610_</small>
        ''', unsafe_allow_html=True)
        st.latex(''' \log xy = \log x + \log y ''')

    elif eqno == 3:
        st.markdown('''
        __3. Calculus__
        \n<small>_Isaac Newton, 1668_</small>
        ''', unsafe_allow_html=True)
        st.latex('''  \\frac{df}{dt} = \lim_{h\\to 0} \\frac{f(t+h) - f(t)}{h}  ''')
        st.latex(''' \int_a^b f(t)dt = F(b) - F(a) ''')

    elif eqno == 4:
        st.markdown('''
        __4. Law of Gravity__
        \n<small>_Isaac Newton, 1687_</small>
        ''', unsafe_allow_html=True)
        st.latex('''  F = G \\frac{m_1m_2}{d^2}  ''')

    elif eqno == 5:
        st.markdown('''
        __5. Black-Scholes Equation__
        \n<small>_Fischer Black, Myron Scholes, 1990_</small>
        ''', unsafe_allow_html=True)
        st.latex('''  \\frac{1}{2} \sigma^2 S^2 \\frac{\partial ^2 V}{\partial S^2} + rS\\frac{\partial V}{\partial S} + \\frac{\partial V}{\partial t} - rV = 0  ''')

    return None


def display_explorer(eqno):

    if eqno == 1:
        pythag_explorer()
    elif eqno == 2:
        log_explorer()
    elif eqno == 3:
        calculus_explorer()
    elif eqno == 4:
        gravity_explorer()
    elif eqno == 5:
        bs_explorer()
    else:
        stub_explorer()

    return None

##########################################################
# stub explorer
##########################################################

def stub_explorer():

    st.markdown('---')

    st.markdown('''
    [Wikipedia](https://en.wikipedia.org/wiki/Calculus):
    This is what it is.

    This is why it's important.
    ''', unsafe_allow_html=True)

    st.markdown('---')

    st.write('under construction...')

    st.markdown('---')

    st.subheader('Further resources')
    st.markdown('''
     - []()
    \n - []()
    \n - []()
    ''', unsafe_allow_html=True)

    return None

##########################################################
# gravity explorer
##########################################################

def gravity_explorer():

    st.markdown('---')

    st.markdown('''
    [Wikipedia's take](https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation):
    Newton's law of universal gravitation states that every point mass attracts every other point mass by a force acting along the line intersecting the two points. The force is proportional to the product of the two masses, and inversely proportional to the square of the distance between them.

    __Our take:__ The discovery is now known as the first great unification, as it marked the first time that the link was proven between what we experience as gravity on Earth, and the force that moves the planets and stars. When an apple falls from a tree, it is experiencing the same force that makes the Moon orbit the Earth.

    This discovery led to an astonishingly deep understanding of the nature of the universe, and enabled space probes and crewed spaceflight, as well as incredibly accurate surveys of Earth, and satellite-enabled navigation systems.
    ''', unsafe_allow_html=True)

    st.markdown('---')

    st.subheader('Universal Gravitational Constant')

    st.latex(''' G = 6.67408 \\times 10^{-11} m^3 kg^{-1} s^{-2} ''')

    st.markdown('---')
    st.subheader('How much would I weigh on the Moon?')
    unit_choice = st.radio('Choose kg or lbs', ['kg', 'lbs'])

    input_string = 'Enter your mass in ' + unit_choice + ':'
    if unit_choice == 'kg': default_val = 70
    else: default_val = 100
    input_mass = st.number_input(input_string, 0, 500, default_val)

    st.write(emoji.emojize(':earth_africa:'), 'On Earth, you weigh ', input_mass, ' ', unit_choice)
    st.write(emoji.emojize(':full_moon:'), 'On the Moon, you would weigh around', round(0.166*input_mass,1), ' ', unit_choice)
    st.write(emoji.emojize(':red_circle:'), 'On Mars, you would weigh around', round(0.377*input_mass,1), ' ', unit_choice)
    st.write(emoji.emojize(':large_blue_circle:'), 'On Gliese 667 Cc, you would weigh around', round(1.6*input_mass,1), ' ', unit_choice)
    st.write(emoji.emojize(':black_circle:'), '400 km away from a basketball-sized black hole, you would weigh around', 3600*input_mass, ' ', unit_choice)

    st.markdown('---')

    st.subheader('Further resources')
    st.markdown('''
    Wikipedia - [Gravity](https://en.wikipedia.org/wiki/Gravity)
    \nKhan Academy - [Introduction to Newton's law of gravitation](https://www.khanacademy.org/science/physics/centripetal-force-and-gravitation/gravity-newtonian/v/introduction-to-newton-s-law-of-gravitation)
    \nJThatch.com - [Gravity Toy](http://jthatch.com/Gravity.js/main.html)
    ''', unsafe_allow_html=True)


    return None



##########################################################
# calculus explorer
##########################################################

def calculus_explorer():

    st.sidebar.subheader('Your equation in x')
    x_cubed = st.sidebar.number_input('x^3', -10, 10, 0, 1)
    x_squared = st.sidebar.number_input('x^2', -10, 10, 1, 1)
    x_first = st.sidebar.number_input('x^1', -10, 10, 0, 1)
    x_const = st.sidebar.number_input('x^0', -10, 10, 0, 1)


    st.markdown('---')

    st.markdown('''
    [Wikipedia's take](https://en.wikipedia.org/wiki/Calculus):
    Calculus is the mathematical study of continuous change, in the same way that geometry is the study of shape and algebra is the study of generalizations of arithmetic operations.
    [John von Neumann](https://en.wikipedia.org/wiki/John_von_Neumann) said that the development of calculus was "the greatest technical advance in exact thinking."

    __Our take:__ Calculus enables scientists to model the real world accurately and usefully, leading to almost all technologies of the modern world, from agriculture to construction, from radio to smartphones, from building simple computers to all modern AI.

    __Differentiation__ gives an estimate of the gradient or rate of change of a variable at a given point.

    __Integration__ is the inverse function of differentiation, giving an estimate of the area under the graph of a given function.
    ''', unsafe_allow_html=True)

    st.markdown('---')

    st.subheader('Your equation')
    st.text('Change with the inputs on the sidebar')
    st.latex('''f(x)=''' + eq_string(0, x_cubed, x_squared, x_first, x_const))
    st.latex('''\\frac{df}{dx} = ''' + eq_string(0, 0, 3*x_cubed, 2*x_squared, x_first))
    st.latex('''\int f(x)dx = ''' + eq_string(x_cubed/4,x_squared/3,x_first/2,x_const,0) + '''+c''')

    st.markdown('---')

    if st.radio('Choose an example', ['Differentiation', 'Integration']) == 'Differentiation':

        st.sidebar.subheader('Differentiation')
        x0 = st.sidebar.slider('x0', -5.0, 5.0, 2.0, 0.1)
        h = st.sidebar.slider('h', 0.01, 3.0, 3.0, 0.01)

        st.subheader('Example - differentiation')
        st.write('Change `h` and `x0` using the sliders')

        st.latex('''  \\frac{df}{dx} = \lim_{h\\to 0} \\frac{f(x+h) - f(x)}{h}  ''')

        generate_diff_chart(h, x0, x_cubed, x_squared, x_first, x_const)

    else:

        st.sidebar.subheader('Integration')
        a = st.sidebar.slider('a', -5.0, 5.0, -4.0, 0.1)
        b = st.sidebar.slider('b', -5.0, 5.0, 4.0, 0.1)
        steps = st.sidebar.slider('steps', 1, 32, 16, 1)

        st.subheader('Example - integration')
        st.write('Change `a`, `b` and `steps` using the sliders')

        st.latex('''\int_a^b f(x)dx \\approx \sum_{i=1}^{steps} w \\times f(a+iw)''')
        st.markdown('''<small>where w represents the width of the approximating 'strips', i.e. b-a divided by the number of steps</small>''', unsafe_allow_html=True)

        generate_int_chart(a, b, steps, x_cubed, x_squared, x_first, x_const)

    st.markdown('---')

    st.subheader('Further resources')
    st.markdown('''
    3blue1brown - [The Essence of Calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM)
    \nMIT Mathematics - [Calculus for beginners](http://www-math.mit.edu/~djk/calculus_beginners/)
    \nMath is Fun - [Calculus](https://www.mathsisfun.com/calculus/index.html)
    ''', unsafe_allow_html=True)

    return None


def generate_int_chart(a, b, steps, a3, a2, a1, a0):

    if a > b: lower_limit, upper_limit = b, a
    else: lower_limit, upper_limit = a, b
    limit_range = upper_limit - lower_limit

    def f_x(x): return a3*x**3 + a2*x**2 + a1*x + a0
    def int_fx(x): return ((a3/4)*x**4 + (a2/3)*x**3 + (a1/2)*x**2 + a0*x)

    x_list = [x/100 for x in range(int(100*(lower_limit-limit_range/10)),int(100*(upper_limit+limit_range/10))+1)]
    y_list = [f_x(x) for x in x_list]

    int_df = pd.DataFrame({'x':x_list, 'f(x)':y_list, 'a':[lower_limit for x in x_list], 'b':[upper_limit for x in x_list], 'x-axis':[0 for x in x_list], 'y-axis':[0 for x in x_list]})

    int_chart_curve = alt.Chart(int_df).mark_line().encode(
            x = alt.X('x:Q', scale=alt.Scale(domain=(int_df.x.min(), int_df.x.max() ) ) ),
            y = alt.Y('f(x):Q', scale=alt.Scale(domain=(min(int_df['f(x)'].min(),0), int_df['f(x)'].max())))
            ).properties(
                title = 'Graph of f(x) and approximate area under the curve'
            ).interactive()

    vert_line_a = alt.Chart(int_df).mark_rule(opacity=0.1, color='red').encode(
            x=alt.X('a:Q', scale=alt.Scale(domain=(int_df.x.min(), int_df.x.max() ) ) )
    ).interactive()

    vert_line_b = alt.Chart(int_df).mark_rule(opacity=0.1, color='red').encode(
            x=alt.X('b:Q', scale=alt.Scale(domain=(int_df.x.min(), int_df.x.max() ) ) )
    ).interactive()

    if lower_limit < 0 and upper_limit > 0: opacity_vert = 0.01
    else: opacity_vert = 0

    if int_df['f(x)'].min() > 0 or int_df['f(x)'].max() < 0: opacity_horiz = 0
    else: opacity_horiz = 0.01

    vert_line_axis = alt.Chart(int_df).mark_rule(opacity=opacity_vert, color='black').encode(
        x=alt.X('y-axis:Q', title='x', scale=alt.Scale(domain=(int_df.x.min(), int_df.x.max() ) ) )
    ).interactive()

    horiz_line_axis = alt.Chart(int_df).mark_rule(opacity=opacity_horiz, color='black').encode(
        y=alt.Y('x-axis:Q', title='f(x)')
    ).interactive()

    centre_bar_list = []
    rect_list_x1 = []
    rect_list_x2 = []
    approx_area = 0
    analytic_area = int_fx(upper_limit) - int_fx(lower_limit)

    for i in range(0, steps):
        x_temp = lower_limit + (limit_range/steps)/2 + i*limit_range/steps
        x_temp_lower = lower_limit + (limit_range/steps) * i
        x_temp_upper = lower_limit + (limit_range/steps) * (i+1)
        centre_bar_list.append(x_temp)
        rect_list_x1.append(x_temp_lower)
        rect_list_x2.append(x_temp_upper)
        approx_area = approx_area + f_x(x_temp) * limit_range/steps

    int_approx_df = pd.DataFrame({'x1':rect_list_x1, 'x2':rect_list_x2, 'y':[f_x(x) for x in centre_bar_list]})

    int_bars = alt.Chart(int_approx_df).mark_rect(filled=True, opacity=.3, color='yellow', stroke='black').encode(
            x=alt.X('x1:Q', scale=alt.Scale(domain=(int_df.x.min(), int_df.x.max() ) ) ),
            x2='x2:Q',
            y='y:Q'
        )

    charts = vert_line_a + vert_line_b + horiz_line_axis + vert_line_axis + int_bars + int_chart_curve
    #charts = int_chart_curve
    charts.layer[0].encoding.y.title = 'f(xadsfdsf)'

    st.altair_chart(charts, use_container_width=True)

    st.markdown('''
    In this case, `steps = {}`, area of the yellow bars = `{}`, exact area = `{}`
    \nAs the number of `steps` increases, the accuracy of the estimate of the area also increases, currently `{}%` difference.
    \nAnalytical integration, where we get an exact expression for the area, is the equation you get when the number of steps approaches infinity.
    '''.format(steps, round(approx_area,4), round(analytic_area,4), round(100*(1-approx_area/analytic_area),2)), unsafe_allow_html=True)

    return None


def generate_diff_chart(h, x0, a3, a2, a1, a0):

    def f_x(x): return a3*x**3 + a2*x**2 + a1*x + a0
    def df_dx(x): return (3*a3*x**2 + 2*a2*x + a1)

    l1_m = df_dx(x0)
    l1_c = f_x(x0)-df_dx(x0)*x0
    l2_m = (f_x(x0+h) - f_x(x0))/h
    l2_c = f_x(x0) - ((f_x(x0+h)-f_x(x0))/h)*x0

    x_list = [x/10 for x in range(-50,85)]
    y_list = [f_x(x) for x in x_list]
    l1_list = [ l1_m*x + l1_c for x in x_list]
    l2_list = [ l2_m*x + l2_c for x in x_list]
    points = [ None for x in x_list]
    points[x_list.index(round(x0,1))] = f_x(x0)
    points[x_list.index(round(x0+h,1))] = f_x(x0+h)
    points_labels = [ None for x in x_list]
    points_labels[x_list.index(round(x0,1))] = 'x0'
    points_labels[x_list.index(round(x0+h,1))] = 'x0 + h'

    diff_df = pd.DataFrame({'x':x_list, 'f(x)':y_list, 'dy/dx':l1_list, 'dy/dx (est)':l2_list, 'points':points, 'points_labels':points_labels, 'x-axis':[0 for x in x_list], 'y-axis':[0 for x in x_list]})

    diff_chart_curve = alt.Chart(pd.melt(diff_df, id_vars=['x'], value_vars=['f(x)', 'dy/dx', 'dy/dx (est)'])).mark_line().encode(
            x = alt.X('x:Q'),
            y = alt.Y('value:Q', scale=alt.Scale(domain=(diff_df['f(x)'].min(), diff_df['f(x)'].max()))),
            color = alt.Color('variable:N', sort=['f(x)', 'dy/dx', 'dy/dx (est)'])
            ).properties(
                title = 'Graph of f(x), dy/dx at x0, and an estimate of dy/dx'
            ).interactive()

    points_chart = alt.Chart(diff_df.drop(['f(x)', 'dy/dx', 'dy/dx (est)'], axis=1)).mark_point().encode(
            x = 'x:Q',
            y = alt.Y('points:Q')
            ).interactive()

    points_labels_chart = alt.Chart(diff_df.drop(['f(x)', 'dy/dx', 'dy/dx (est)'], axis=1)).mark_text(
            align='center',
            dx=0,
            dy=-10
    ).encode(
            x = 'x:Q',
            y = alt.Y('points:Q'),
            text='points_labels'
    ).interactive()

    vert_line_axis = alt.Chart(diff_df).mark_rule(opacity=0.05, color='gray').encode(
        x=alt.X('y-axis:Q', title='x')
    ).interactive()

    horiz_line_axis = alt.Chart(diff_df).mark_rule(opacity=0.05, color='gray').encode(
        y=alt.Y('x-axis:Q', title='f(x)')
    ).interactive()


    charts = vert_line_axis + horiz_line_axis + diff_chart_curve + points_chart + points_labels_chart

    st.markdown('''
    In the chart below, you can see that as h gets smaller, the estimate of the gradient at x0 gets closer to the actual value.
    ''')

    st.markdown('''
    x<sub>0</sub> `{}` f(x<sub>0</sub>) `{}` h `{}` dy/dx `{}` dy/dx (est) `{}`
    '''.format(x0, round(f_x(x0),2), h, round(l1_m,2), round(l2_m,2)), unsafe_allow_html=True)
    st.write('')

    st.altair_chart(charts, use_container_width=True)

    return None


def eq_string(a4, a3, a2, a1, a0):

    if float(a4).as_integer_ratio()[1] != 1:
        a4_str_temp = '\\frac{'+str(abs(Fraction(a4).limit_denominator())).split('/')[0]+'}{'+str(Fraction(a4).limit_denominator()).split('/')[1]+'}'
    else:
        a4_str_temp = str(int(abs(a4)))

    if float(a3).as_integer_ratio()[1] != 1:
        a3_str_temp = '\\frac{'+str(abs(Fraction(a3).limit_denominator())).split('/')[0]+'}{'+str(Fraction(a3).limit_denominator()).split('/')[1]+'}'
    else:
        a3_str_temp = str(int(abs(a3)))

    if float(a2).as_integer_ratio()[1] != 1:
        a2_str_temp = '\\frac{'+str(abs(Fraction(a2).limit_denominator())).split('/')[0]+'}{'+str(Fraction(a2).limit_denominator()).split('/')[1]+'}'
    else:
        a2_str_temp = str(int(abs(a2)))

    if a4 == 0: a4_str = ''
    elif a4 == -1: a4_str = '-x^4'
    elif a4 == 1: a4_str = 'x^4'
    elif a4 < 0: a4_str = '-' + a4_str_temp + 'x^4'
    else: a4_str = a4_str_temp + 'x^4'

    if a3 == 0: a3_str = ''
    elif a3 == -1: a3_str = '- x^3'
    elif a3 == 1:
        if a4 != 0: a3_str = '+ x^3'
        else: a3_str = 'x^3'
    elif a3 > 0 and a4 != 0: a3_str = '+' + a3_str_temp + 'x^3'
    elif a3 < 0: a3_str = '-' + a3_str_temp + 'x^3'
    else: a3_str = a3_str_temp +'x^3'

    if a2 == 0: a2_str = ''
    elif a2 == -1: a2_str = '-x^2'
    elif a2 == 1:
        if (a4 != 0 or a3 !=0): a2_str = '+ x^2'
        else: a2_str = 'x^2'
    elif a2 > 0 and (a4 != 0 or a3 !=0): a2_str = '+' + a2_str_temp + 'x^2'
    elif a2 < 0: a2_str = '-' + a2_str_temp + 'x^2'
    else: a2_str = a2_str_temp +'x^2'

    if a1 == 0: a1_str = ''
    elif a1 == -1: a1_str = '-x'
    elif a1 == 1:
        if (a4 != 0 or a3 != 0 or a2 != 0): a1_str = '+x'
        else: a1_str = 'x'
    elif a1 > 0 and (a4 != 0 or a3 != 0 or a2 != 0): a1_str = '+' + str(a1) + 'x'
    else: a1_str = str(a1)+'x'

    if a0 == 0: a0_str = ''
    elif a0 > 0 and (a4 != 0 or a3 != 0 or a2 != 0 or a1 != 0): a0_str = '+' + str(a0)
    else: a0_str = str(a0)

    if a4 == a3 == a2 == a1 == a0 == 0: e_str = '0'
    else: e_str = a4_str + a3_str + a2_str + a1_str + a0_str

    return e_str



##########################################################
# log explorer
##########################################################

def log_explorer():

    x = st.sidebar.slider('x', 0, 100, 27, 1)
    y = st.sidebar.slider('y', 0, 100, 34, 1)

    common_bases = st.sidebar.checkbox('Common bases', value=True)

    if common_bases:
        base_jump = st.sidebar.selectbox('Common bases', ['2','e','10'], 2)
        if base_jump == '2': base = 2
        elif base_jump == 'e': base = math.e
        elif base_jump == '10': base = 10
        else: st.error('Base not found')
    else:
        base = st.sidebar.slider('base', 0.1, 10.0, 3.0, 0.2)

#### Wiki explanation

    st.markdown('---')

    st.markdown('''
    [Wikipedia's take](https://en.wikipedia.org/wiki/Logarithm):
    In mathematics, the logarithm is the inverse function to exponentiation.
    That means the logarithm of a given number x is the exponent to which another fixed number, the base b, must be raised, to produce that number x.
    In the simplest case, the logarithm counts the number of occurrences of the same factor in repeated multiplication; e.g., since 1000 = 10 × 10 × 10 = 10<sup>3</sup>, the "logarithm base 10" of 1000 is 3, or log<sub>10</sub>(1000) = 3.

    __Our take:__ Enabling complex calcuations quickly, hundreds of years before the advent of computers, meant advances in many fields, and today logarithms are everywhere:
    computer science; information theory; music theory; photography; mathematics; physics; chemistry; statistics; economics; and engineering.

    The Apollo 11 mission that landed people on the moon was only made possible with logarithms: NASA scientists used slide rules to design the rockets and trajectories needed, and the Apollo astronauts all took slide rules with them on their journeys to the Moon.

    ''', unsafe_allow_html=True)

    st.markdown('---')

    st.subheader('Example - use the sliders to change the values')

    st.write('Base: ', round(base,1))
    st.write('x = ', x, 'log x = ', round(math.log(x, base), 3))
    st.write('y = ', y, 'log y = ', round(math.log(y, base), 3))
    st.write('x*y = ', x*y, 'log (xy) = ', round(math.log(x*y, base), 3))
    st.markdown('You can see that multiplying `x` by `y` might be difficult, but adding `log(x)` to `log(y)` is easy.')
    st.markdown('''
    This enabled people to do complicated calculations much more quickly, with look-up tables of logarithms.
    For example, to multiply two large numbers, `x` and `y`:
    * look up `log(x)`
    * look up `log(y)`
    * add `log(x)` to `log(y)`
    * look up inverse log of this sum

    Multiplication (as above), division, exponentiation, and taking the n<sup>th</sup> root of large numbers all became possible, quickly, with logarithm look-up tables (published in large pamphlets or books), and later, the slide rule.
    ''', unsafe_allow_html=True)

    st.markdown('---')

    x_log_list = [(n+1)/10 for n in range(1000)]
    y_log_half_list = [math.log(item, 0.5) for item in x_log_list]
    y_log_2_list = [math.log(item, 2) for item in x_log_list]
    y_log_e_list = [math.log(item, math.e) for item in x_log_list]
    y_log_10_list = [math.log(item, 10) for item in x_log_list]
    if common_bases:
        log_df = pd.melt(pd.DataFrame({'x':x_log_list, 'base 1/2':y_log_half_list, 'base 2':y_log_2_list, 'base e':y_log_e_list, 'base 10':y_log_10_list}), id_vars=['x'], value_vars=['base 1/2', 'base 2', 'base e', 'base 10'])
    else:
        y_log_base_list = [math.log(item, base) for item in x_log_list]
        log_df = pd.melt(pd.DataFrame({'x':x_log_list, 'base 1/2':y_log_half_list, 'base 2':y_log_2_list, 'base e':y_log_e_list, 'base 10':y_log_10_list, 'base '+str(base):y_log_base_list}), id_vars=['x'], value_vars=['base 1/2', 'base 2', 'base e', 'base 10', 'base '+str(base)])

    y_exp_base_list = [base**item for item in x_log_list[0:25]]
    exp_df = pd.DataFrame({'x':x_log_list[0:25], 'y':y_exp_base_list})


    log_chart = alt.Chart(log_df).mark_line().encode(
        x = 'x:Q',
        y = alt.Y('value:Q', scale=alt.Scale(type='linear')),
        color = alt.Color('variable:N', sort=['base 1/2', 'base 2', 'base e', 'base 10', 'base '+str(base)])
    ).interactive()

    exp_chart = alt.Chart(exp_df).mark_line().encode(
        x = 'x:Q',
        y = alt.Y('y:Q', scale=alt.Scale(type='linear')),
    ).properties(
        width = 250
    ).interactive()

    log_exp_chart = alt.Chart(exp_df).mark_line().encode(
        x = 'x:Q',
        y = alt.Y('y:Q', scale=alt.Scale(type='log')),
    ).properties(
        width = 250
    ).interactive()

    st.subheader('Graph of logarithms')
    st.altair_chart(log_chart, use_container_width=True)

    if round(base, 4) == round(math.e, 4): base_to_display = 'e'
    else: base_to_display = base

    st.subheader('Plotting data with exponential growth')
    st.markdown('''
    y = {} <sup> x </sup>
    \n<small>You can see that the y axis scale grows as fast as the exponential data, making it a straight line</small>
    '''.format(base_to_display), unsafe_allow_html=True)
    st.altair_chart(exp_chart | log_exp_chart)

    st.markdown('---')

    st.subheader('Further resources')
    st.markdown('''
    Math is Fun - [Introduction to Logarithms](https://www.mathsisfun.com/algebra/logarithms.html)
    \nNancyPi - [Logarithms... How?](https://www.youtube.com/watch?v=Zw5t6BTQYRU)
    \nPurple Math - [Logarithms: Introduction to "The Relationship"](https://www.purplemath.com/modules/logs.htm)
    ''', unsafe_allow_html=True)

    return None


##########################################################
# Pythagoras' theorem explorer
##########################################################

def pythag_explorer():

#### Sidebar user choices

    st.sidebar.text('Change the sizes of the sides')

    max_ab = st.sidebar.slider('Maximum of a and b', 10, 100, 20, 1)

    a = st.sidebar.slider('a', 1, max_ab, 4, 1)
    b = st.sidebar.slider('b', 1, max_ab, 3, 1)

#### Wiki explanation

    st.markdown('---')

    st.markdown('''
    [Wikipedia's take](https://en.wikipedia.org/wiki/Pythagorean_theorem):
    Pythagoras' theorem, is a fundamental relation in Euclidean geometry among the three sides of a right triangle. It states that the area of the square whose side is the hypotenuse (the side opposite the right angle) is equal to the sum of the areas of the squares on the other two sides. This theorem can be written as an equation relating the lengths of the sides a, b and c, often called the "Pythagorean equation".

    __Our take:__ It links geometry and algebra in a fundamental way, and generalisations of Pythagoras' led to trigonometry, our ability to survey land, navigate the seas, and measure the universe itself.
    ''', unsafe_allow_html=True)

#### Explorer element - take the slider values and generate a triangle

    st.markdown('---')

    st.subheader('''Example - use the sliders to change the triangle below''')
    generate_triangle_svg(a, b)

    st.markdown('---')

    st.subheader('''Rearrangement proof of Pythagoras' theorem''')
    st.markdown('''
    [Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_theorem):
    The two large squares shown in the figure each contain four identical triangles, and the only difference between the two large squares is that the triangles are arranged differently. Therefore, the white space within each of the two large squares must have equal area. Equating the area of the white space yields the Pythagorean theorem, Q.E.D.
    ''', unsafe_allow_html=True)

    generate_rearr_proof_svg(a, b)

########### ->

    st.markdown('---')

    st.subheader('''Pythagorean triples''')
    st.markdown('''
    [Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_triple):
    A Pythagorean triple consists of three positive integers a, b, and c, such that a<sup>2</sup> + b<sup>2</sup> = c<sup>2</sup>. Such a triple is commonly written (a, b, c), and a well-known example is (3, 4, 5).
    ''', unsafe_allow_html=True)

    triple_gen_button = st.button('Click to generate a triple')

    triple_gen_placeholder = st.empty()

    if triple_gen_button:
        i, j, k = generate_triple(a, b)
        triple_gen_placeholder.markdown('''
        `({},{},{})`
        \n{}<sup>2</sup> = `{}`
        \n{}<sup>2</sup> = `{}`
        \n{}<sup>2</sup> = `{}`
        '''.format(i, j, k, i, i**2, j, j**2, k, k**2), unsafe_allow_html=True)

    st.markdown('---')

    st.subheader('''Further resources''')
    st.markdown('''
    [3blue1brown](https://www.youtube.com/watch?v=QJYmyhnaaek)
    \n[Wolfram Demonstrations Project](https://demonstrations.wolfram.com/PlottingPythagoreanTriples/)
    \n[Wolfram MathWorld](https://mathworld.wolfram.com/PythagoreanTheorem.html)
    ''', unsafe_allow_html=True)

    return None


def generate_triple(a, b):
    # Based on 6:54 in https://www.youtube.com/watch?v=QJYmyhnaaek

    if a < b: u, v = b, a
    elif b > a: u, v = a, b
    else: u, v = a+1, b

    x_ = u**2 - v**2
    y_ = 2*u*v
    z = u**2 + v**2

    if x_ < y_: x, y = x_, y_
    else: x, y = y_, x_

    # Fallback triple list to return
    triple_list = [(3, 4, 5), (5, 12, 13), (7, 24, 25), (8, 15, 17), (9, 40, 41),
               (11, 60, 61), (12, 35, 37), (13, 84, 85), (15,112,113), (16, 63, 65),
               (17,144,145), (19,180,181), (20, 21, 29), (20, 99,101), (21,220,221),
               (23,264,265), (24,143,145), (25,312,313), (27,364,365), (28, 45, 53),
               (28,195,197), (29,420,421), (31,480,481), (32,255,257), (33, 56, 65),
               (33,544,545), (35,612,613), (36, 77, 85), (36,323,325), (37,684,685)]

    return x,y,z


def generate_triangle_svg(base, height):

    a, b, c = base, height, math.sqrt(base**2+height**2)

    if a > b:
        scaling_factor = 400 / (2*a+b)
    else:
        scaling_factor = 400 / (a+2*b)

    #scaling_factor = 3

    black_x, black_y, black_width, black_height = 340-(a*scaling_factor)/2, 250+(b*scaling_factor)/2, a * scaling_factor, a * scaling_factor
    yellow_x, yellow_y, yellow_width, yellow_height = 340+(a*scaling_factor)/2, 250-(b*scaling_factor)/2, b * scaling_factor, b * scaling_factor

    theta_deg = -1*math.atan(b/a)*180/math.pi
    theta_rad = -1*math.atan(b/a)

    x = (340 - a*scaling_factor/2 + (250+b*scaling_factor/2)*math.tan(theta_rad))/(math.cos(theta_rad) + math.sin(theta_rad)*math.tan(theta_rad))
    y = (250 + b*scaling_factor/2 - x*math.sin(theta_rad))/math.cos(theta_rad) - c*scaling_factor

    pink_x, pink_y, pink_width, pink_height, pink_rotation = x, y, c * scaling_factor, c * scaling_factor, theta_deg

    svg = '''
    <svg xmlns="http://www.w3.org/2000/svg" width="680" height="500" viewBox="0 0 680 500"><g>
        <rect id="rect_a_black" style="fill:#262730;fill-opacity:1;stroke-width:0" x="{}" y="{}" height="{}" width="{}" />
        <rect id="rect_b_yellow" style="fill:#fffd80;fill-opacity:1;stroke-width:0" x="{}" y="{}" height="{}" width="{}" />
        <rect id="rect_c_pink" style="fill:#f63366;fill-opacity:1;stroke-width:0" x="{}" y="{}" height="{}" width="{}" transform="rotate({})" />
        <polygon points="{}" style="stroke:#000000;fill:#f63366;fill-opacity:.1;stroke-width:0" />

        <text x="0" y="100" fill="black" font-family="monospace"> a = {} </text>
        <text x="0" y="125" fill="black" font-family="monospace"> a<tspan baseline-shift="super">2</tspan> = {} </text>
        <text x="0" y="175" fill="black" font-family="monospace"> b = {} </text>
        <text x="0" y="200" fill="black" font-family="monospace"> b<tspan baseline-shift="super">2</tspan> = {} </text>
        <text x="0" y="250" fill="black" font-family="monospace"> c = {} </text>
        <text x="0" y="275" fill="black" font-family="monospace"> c<tspan baseline-shift="super">2</tspan> = {}</text>

        <text x="{}" y="{}" fill="black" font-family="monospace">a</text>
        <text x="{}" y="{}" fill="black" font-family="monospace">b</text>
        <text x="{}" y="{}" fill="black" font-family="monospace">c</text>

        <text x="{}" y="{}" fill="black" font-family="monospace">Scale</text>
        <line x1="{}" y1="{}" x2="{}" y2="{}" style="stroke:rgb(0,0,0);stroke-width:1" />
        <line x1="{}" y1="{}" x2="{}" y2="{}" style="stroke:rgb(0,0,0);stroke-width:1" />
        <line x1="{}" y1="{}" x2="{}" y2="{}" style="stroke:rgb(0,0,0);stroke-width:1" />
        <text x="{}" y="{}" fill="black" font-family="monospace">0</text>
        <text x="{}" y="{}" fill="black" font-family="monospace">{}</text>
    </g></svg>
    '''.format(black_x, black_y, black_width, black_height,
               yellow_x, yellow_y, yellow_width, yellow_height,
               pink_x, pink_y, pink_width, pink_height, pink_rotation,
               str(black_x)+','+str(black_y)+' '+str(yellow_x)+','+str(yellow_y)+' '+str(yellow_x)+','+str(black_y),
               a, a**2, b, b**2, round(c,2), a**2 + b**2,
               black_x+black_width/2, black_y-5,
               yellow_x-10, yellow_y + yellow_height/2+10,
               black_x+black_width/2, yellow_y + yellow_height/2+10,
               0, 420,
               0, 430, a*scaling_factor, 430,
               1, 425, 1, 435,
               a*scaling_factor-5, 425, a*scaling_factor-5, 435,
               0, 450,
               a*scaling_factor-10, 450, a
               )

    render_svg(svg)

    return None

def generate_rearr_proof_svg(base, height):

    rebase = int(250 * base / (base + height))
    reheight = int(250 * height / (base + height))

    color_bottom_left = '262730fffd80'
    color_top_right = 'fffd80'


    svg = '''
    <svg xmlns="http://www.w3.org/2000/svg" width="680" height="270" viewBox="0 0 680 270"><g>
        <rect id="rect_left" style="stroke:#000000;fill:#f63366;fill-opacity:1;stroke-width:0" x="{}" y="{}" height="{}" width="{}" />
        <rect id="rect_right" style="stroke:#000000;fill:#ffffff;fill-opacity:.1;stroke-width:0" x="{}" y="{}" height="{}" width="{}" />
        <polygon points="{}" style="stroke:#000000;fill:#ffffff;fill-opacity:.9;stroke-width:0" />
        <polygon points="{}" style="stroke:#000000;fill:#ffffff;fill-opacity:.85;stroke-width:0" />
        <polygon points="{}" style="stroke:#000000;fill:#ffffff;fill-opacity:.8;stroke-width:0" />
        <polygon points="{}" style="stroke:#000000;fill:#ffffff;fill-opacity:.75;stroke-width:0" />
        <polygon points="{}" style="stroke:#000000;fill:#f63366;fill-opacity:.1;stroke-width:0" />
        <polygon points="{}" style="stroke:#000000;fill:#f63366;fill-opacity:.15;stroke-width:0" />
        <polygon points="{}" style="stroke:#000000;fill:#f63366;fill-opacity:.2;stroke-width:0" />
        <polygon points="{}" style="stroke:#000000;fill:#f63366;fill-opacity:.25;stroke-width:0" />
        <rect style="stroke:#000000;fill:#{};fill-opacity:1;stroke-width:0" x="{}" y="{}" height="{}" width="{}" />
        <rect style="stroke:#000000;fill:#{};fill-opacity:1;stroke-width:0" x="{}" y="{}" height="{}" width="{}" />
        <text x="{}" y="{}" fill="black" font-family="monospace"> c<tspan baseline-shift="super">2</tspan> </text>
        <text x="{}" y="{}" fill="white" font-family="monospace"> a<tspan baseline-shift="super">2</tspan> </text>
        <text x="{}" y="{}" fill="black" font-family="monospace"> b<tspan baseline-shift="super">2</tspan> </text>
    </g></svg>
    '''.format(70, 10, 250, 250,
               360, 10, 250, 250,
               '70,10 70,'+str(reheight+10)+' '+str(rebase+70)+',10',
               str(rebase+70)+',10 320,10 320,'+str(rebase+10),
               '320,'+str(rebase+10)+' 320,260 '+str(70+reheight)+',260',
               str(70+reheight)+',260 70,260 70,'+str(10+reheight),
               '360,10 '+str(360+reheight)+',10 360,'+str(10+rebase),
               str(360+reheight)+',10, 360,'+str(10+rebase)+' '+str(360+reheight)+','+str(10+rebase),
               str(360+reheight)+','+str(10+rebase)+' '+str(360+reheight)+',260 610,260',
               str(360+reheight)+','+str(10+rebase)+' 610,260 610,'+str(10+rebase),
               color_bottom_left, 360+reheight, 10, 250-reheight, 250-reheight,
               color_top_right, 360, 10+rebase, 250-rebase, 250-rebase,
               70+125-5, 10+125+5,
               360+reheight+rebase/2 - 5, 10+rebase/2 + 5,
               360+reheight/2 - 5, 10+rebase+reheight/2 + 5,
               )


    render_svg(svg)

    return None


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

    return html




##########################################################
# BS explorer
##########################################################

def bs_explorer():

    # Option paramaters

    st.sidebar.text('European vanilla option parameters')
    T_t = st.sidebar.slider('Time to maturity (years)', 0.0,10.0,2.0,0.5, format='%.1f')
    S_t = st.sidebar.slider('Current spot price (USD)',10,250,100,10)
    K = st.sidebar.slider('Option strike price (USD)',10,250,100,10)
    r = st.sidebar.slider('Risk-free rate', -0.05, 0.2, 0.03, 0.01, format='%.2f')
    #r = st.sidebar.slider('Risk-free rate', -5%, 20%, 3%, 1%, format='%.2f')
    sigma = st.sidebar.slider('Vol of underlying (annualised)', 0.0, 1.0, 0.2, 0.01)

    c_val, p_val = bs_pricer(T_t, S_t, K, r, sigma)

    c_delta, c_theta, c_rho, p_delta, p_theta, p_rho, gamma, vega = bs_greeks(T_t, S_t, K, r, sigma)


    # Create DataFrame
    option_data = {'Value':['Value (USD)', 'delta', 'gamma', 'vega', 'theta', 'rho'],
                   'Call':[c_val, c_delta, gamma, vega, c_theta, c_rho],
                   'Put':[p_val, p_delta, gamma, vega, p_theta, p_rho]}

    option_df = pd.DataFrame(option_data).set_index('Value')
    option_df['Call'] = option_df['Call'].map('{:,.2f}'.format)
    option_df['Put'] = option_df['Put'].map('{:,.2f}'.format)

    spot_list = np.arange(5, 255, 5)
    put_name_list = ['Put' for i in range(len(spot_list))]
    call_name_list = ['Call' for i in range(len(spot_list))]

    call_value_list, put_value_list = [], []
    c_delta_list, c_theta_list, c_rho_list, p_delta_list, p_theta_list, p_rho_list, gamma_list, vega_list = [], [], [], [], [], [], [], []

    for i in range(len(spot_list)):
        cva, pva = bs_pricer(T_t, spot_list[i], K, r, sigma)
        call_value_list.append(cva)
        put_value_list.append(pva)

        c_d_a, c_t_a, c_r_a, p_d_a, p_t_a, p_r_a, g_a, v_a = bs_greeks(T_t, spot_list[i], K, r, sigma)
        c_delta_list.append(c_d_a)
        c_theta_list.append(c_t_a)
        c_rho_list.append(c_r_a)
        p_delta_list.append(p_d_a)
        p_theta_list.append(p_t_a)
        p_rho_list.append(p_r_a)
        gamma_list.append(g_a)
        vega_list.append(v_a)

    alternative_osd_c = pd.DataFrame({'C-P':call_name_list,
                                    'Spot':spot_list.tolist(),
                                    'Valuation':call_value_list,
                                    'Delta':c_delta_list,
                                    'Theta':c_theta_list,
                                    'Rho':c_rho_list,
                                    'Gamma':gamma_list,
                                    'Vega':vega_list
                                   })

    alternative_osd_p = pd.DataFrame({'C-P':put_name_list,
                                    'Spot':spot_list.tolist(),
                                    'Valuation':put_value_list,
                                    'Delta':p_delta_list,
                                    'Theta':p_theta_list,
                                    'Rho':p_rho_list,
                                    'Gamma':gamma_list,
                                    'Vega':vega_list
                                   })

    alternative_osd = pd.melt(alternative_osd_c.append(alternative_osd_p), id_vars = ['C-P', 'Spot'], value_vars = ['Valuation', 'Delta', 'Gamma', 'Vega', 'Rho', 'Theta'])

    valuation_chart = alt.Chart(alternative_osd[(alternative_osd['variable']=='Valuation')]).mark_line().encode(
        x = 'Spot:Q',
        y = alt.Y('value:Q', axis=alt.Axis(title=None)),
        color = alt.Color('variable:N', legend=None)
    ).properties(
        width = 250,
        height = 100
    ).facet(
        column = alt.Column('C-P:N', title=None),
        row = alt.Row('variable:N', title=None)
    ).resolve_scale(y='independent')

    greeks_chart = alt.Chart(alternative_osd[(alternative_osd['variable']!='Valuation')]).mark_line().encode(
        x = 'Spot:Q',
        y = alt.Y('value:Q', axis=alt.Axis(title='')),
        color = alt.Color('variable:N', legend=None)
    ).properties(
        width = 250,
        height = 100
    ).facet(
        column = alt.Column('C-P:N', title=None),
        row = alt.Row('variable:N', title=None)
    ).resolve_scale(y='independent')

    st.markdown('---')

    st.markdown('''
    [Wikipedia's take](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model):
    The Black–Scholes model is a mathematical model for the dynamics of a financial market containing derivative investment instruments.
    From the partial differential equation in the model, known as the Black–Scholes equation, one can deduce the Black–Scholes formula, which gives a theoretical estimate of the price of European-style options.

    __Our take:__ Having a commonly agreed model in place for the valuation of derivatives enabled efficient markets in these instruments, which in turn allowed for cheaper and better financial risk management for companies, enabling far greater levels of international trade and economic activity.
    This in turn led to lifting hundreds of millions of people out of poverty.
    ''', unsafe_allow_html=True)

    st.markdown('---')

    st.subheader('Example - use the sliders to change the values')

    st.markdown('''
    Value of option with spot {}, strike {}, {} yrs to run, rfr of {}%, and vol of {}% is:
    '''.format(S_t, K, T_t, r*100, sigma*100), unsafe_allow_html=True)
    st.write(option_df.transpose().drop(['delta', 'gamma', 'vega', 'theta', 'rho'], axis=1))

    st.altair_chart(valuation_chart)

    st.markdown('---')

    st.subheader('The Greeks')
    st.markdown('''
    [Wikipedia](https://en.wikipedia.org/wiki/Greeks_(finance)):
    Once the valuation of the option has been calculated, Black–Scholes enables the derivation of 'the greeks',
    which are the quantities representing the sensitivity of the price of the option to a change in underlying parameters.
    The greeks are also known as risk sensitivities, risk measures or hedge parameters.
    ''', unsafe_allow_html=True)

    st.write(option_df.transpose())

    st.altair_chart(greeks_chart)

    st.markdown('---')

    st.subheader('Further resources')
    st.markdown('''
    Corporate Finance Institute - [BSM](https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/black-scholes-merton-model/)
    \nInvestopedia - [Black Scholes Model](https://www.investopedia.com/terms/b/blackscholes.asp)
    \nColumbia University - [The Black-Scholes Model](http://www.columbia.edu/~mh2078/FoundationsFE/BlackScholes.pdf)
    ''', unsafe_allow_html=True)

    return None



def bs_pricer(T_t, S_t, K, r, sigma):
    ''' Returns call and put value given t, S, K, r, sigma '''

    d1 = (1 / (sigma * math.sqrt(T_t))) * ( math.log(S_t/K) + (r + 0.5 * sigma ** 2) * T_t )
    d2 = d1 - (sigma * math.sqrt(T_t))
    pvk = K * math.exp(-1*r*T_t)

    call_val = sstats.norm.cdf(d1) * S_t - sstats.norm.cdf(d2) * pvk
    put_val = K * math.exp(-1*r*T_t) - S_t + call_val

    call_delta = sstats.norm.cdf(d1)
    call_gamma = sstats.norm.pdf(d1) / (S_t * sigma * math.sqrt(T_t))
    call_vega = S_t * sstats.norm.pdf(d1) * math.sqrt(T_t)
    call_theta = -1 * (S_t * sstats.norm.pdf(d1) * sigma / (2 * math.sqrt(T_t))) - r * K * math.exp(-1*r*T_t) * sstats.norm.cdf(d2)
    call_rho = K * T_t * math.exp(-1*r*T_t) * sstats.norm.cdf(d2)

    put_delta = call_delta - 1
    put_gamma = call_gamma
    put_vega = call_vega
    put_theta = -1 * (S_t * sstats.norm.pdf(d1) * sigma / (2 * math.sqrt(T_t))) + r * K * math.exp(-1*r*T_t) * sstats.norm.cdf(-1*d2)
    put_rho = -1* K * T_t * math.exp(-1*r*T_t) * sstats.norm.cdf(-1* d2)

    return call_val, put_val


def bs_greeks(T_t, S_t, K, r, sigma):
    ''' Returns call and put greeks given t, S, K, r, sigma '''

    d1 = (1 / (sigma * math.sqrt(T_t))) * ( math.log(S_t/K) + (r + 0.5 * sigma ** 2) * T_t )
    d2 = d1 - (sigma * math.sqrt(T_t))
    pvk = K * math.exp(-1*r*T_t)

    call_val = sstats.norm.cdf(d1) * S_t - sstats.norm.cdf(d2) * pvk
    put_val = K * math.exp(-1*r*T_t) - S_t + call_val

    call_delta = sstats.norm.cdf(d1)
    call_theta = -1 * (S_t * sstats.norm.pdf(d1) * sigma / (2 * math.sqrt(T_t))) - r * K * math.exp(-1*r*T_t) * sstats.norm.cdf(d2)
    call_rho = K * T_t * math.exp(-1*r*T_t) * sstats.norm.cdf(d2)

    put_delta = call_delta - 1
    put_theta = -1 * (S_t * sstats.norm.pdf(d1) * sigma / (2 * math.sqrt(T_t))) + r * K * math.exp(-1*r*T_t) * sstats.norm.cdf(-1*d2)
    put_rho = -1* K * T_t * math.exp(-1*r*T_t) * sstats.norm.cdf(-1* d2)

    # Note that gamma and vega are symmetrical, ie the same for puts and calls

    gamma = sstats.norm.pdf(d1) / (S_t * sigma * math.sqrt(T_t))
    vega = S_t * sstats.norm.pdf(d1) * math.sqrt(T_t)

    return call_delta, call_theta, call_rho, put_delta, put_theta, put_rho, gamma, vega




##########################################################
# MAIN()
##########################################################

def main():

    st.sidebar.header('Equations explorer')

    se_list = ['Display all equations',
                '1. Pythagoras\' Theorem',
                '2. Logarithms',
                '3. Calculus',
                '4. Law of Gravity',
                '5. Black-Scholes Equation']

    chosen_eq = st.sidebar.selectbox('Choose your equation', se_list)

    if chosen_eq == 'Display all equations':

        st.header('Equations that changed the world')
        st.markdown('''
        Based on the compilation in the 2012 [book](https://en.wikipedia.org/wiki/In_Pursuit_of_the_Unknown) by Ian Stewart
        ''', unsafe_allow_html=True)
        st.subheader('')

        for i in range(len(se_list)):
            write_eq(i)
    else:
        write_eq(se_list.index(chosen_eq))
        display_explorer(se_list.index(chosen_eq))

    return None


if __name__ == '__main__':
    main()
