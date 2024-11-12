import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import math

# Funkcja równań różniczkowych modelu maszyny indukcyjnej
def rownania_maszyny_induk(t, y0, a1, a2, a3, a4, a5, a6, a7, omega_r, u_s, J):
    
    Isa, Isb, psi_ra, psi_rb, domega_r_dt = y0
    mo = 0.1
    # Równania różniczkowe
    dis_a_dt = a1 * Isa + a2 * psi_ra - a3 * omega_r * psi_rb + a4 * abs(u_s)
    dis_b_dt = a1 * Isb + a2 * psi_rb + a3 * omega_r * psi_ra + a4 * abs(u_s)

    dpsi_ra_dt = a5 * Isa + a6 * psi_ra - omega_r * psi_rb
    dpsi_rb_dt = a5 * Isb + a6 * psi_rb + omega_r * psi_ra

    domega_r_dt = 1/J*(a7 * (psi_ra * Isb - psi_rb * Isa) - mo)
    return np.array([dis_a_dt, dis_b_dt, dpsi_ra_dt, dpsi_rb_dt, domega_r_dt])

# Funkcja RK4 dla układu równań różniczkowych
def runge_kutta(f, y0, t_end, h, *args):
    t = np.arange(0, t_end, h)
    y = np.zeros((len(t), len(y0)))  # Macierz na wyniki 
    
    y[0] = y0  # Ustawienie warunków początkowych
    
    for i in range(1, len(t)):
        # Obliczanie wartości u_s w każdym kroku
        # foc_control()
        u_s = args[-2] * np.sin(2 * np.pi * args[-1] * t[i-1])
        
        # Argumenty bez u_s i częstotliwości f
        rk_args = args[:-2] + (u_s,)
        
        k1 = h * f(t[i-1], y[i-1], *rk_args)
        k2 = h * f(t[i-1] + 0.5*h, y[i-1] + 0.5*k1, *rk_args)
        k3 = h * f(t[i-1] + 0.5*h, y[i-1] + 0.5*k2, *rk_args)
        k4 = h * f(t[i-1] + h, y[i-1] + k3, *rk_args)
        # 1/6 = 0.166666
        y[i] = y[i-1] + 0.166666 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

# Regulator PI
def pi_controller(error, kp, ki, integral, dt):
    integral += error * dt
    output = kp * error + ki * integral
    return output, integral

# Główna funkcja sterowania FOC
def foc_control(i_a, i_b, i_c, theta, i_d_ref, i_q_ref, kp_d, ki_d, kp_q, ki_q, dt):
    # Transformacja Clarka i Parka
    i_alpha, i_beta = transformata_clarka(i_a, i_b, i_c)
    i_d, i_q = transformata_parka(i_alpha, i_beta, theta)
    
    # Regulacja prądu d (kontrola strumienia)
    error_d = i_d_ref - i_d
    v_d, integral_d = pi_controller(error_d, kp_d, ki_d, 0, dt)

    # Regulacja prądu q (kontrola momentu)
    error_q = i_q_ref - i_q
    v_q, integral_q = pi_controller(error_q, kp_q, ki_q, 0, dt)
    
    # Transformacja odwrotna Parka i Clarka
    v_alpha, v_beta = odwrocona_transformata_parka(v_d, v_q, theta)
    
    return v_alpha, v_beta

# Funkcja obliczajaca moduł wektorow pradu i strumienia
def mod(t, y):
    modules = np.zeros((len(t),2))
    for i in range(1, len(t)):
        modules[i-1][0] = math.sqrt(y[i-1][0]**2 + y[i-1][1]**2)
        modules[i-1][1] = math.sqrt(y[i-1][2]**2 + y[i-1][3]**2) 
    return t, modules

def transformata_clarka(x, y):
    x1 = x
    y2 = (2 / np.sqrt(3)) * (x * np.sin(np.pi/3) - y * np.cos(np.pi/3))
    return x1, y2

def transformata_parka(x, y, fi):
    x1 = x * np.cos(fi) + y * np.sin(fi)
    y1 = -x * np.sin(fi) + y * np.cos(fi)
    return x1, y1

def odwrocona_transformata_parka(x, y, fi):
    x1 = x * np.cos(fi) - y * np.sin(fi)
    y1 = x * np.sin(fi) + y * np.cos(fi)
    return x1, y1

def odwrocona_transformata_clarka(x, y):
    x1 = x
    y1 = -0.5 * x + (np.sqrt(3)/2) * y
    return x1, y1

# Funkcja do rysowania wyników na kilku wykresach
def toDo(t, y, y_2):
    fig, axs = plt.subplots(6, 1, figsize=(8, 6))  # 6 wykresów w jednym oknie
    
    # Wykres dla Is (prąd)
    axs[0].plot(t, y[:, 0], label='Isa (prąd)', color='r')
    axs[0].set_xlabel('Czas (s)')
    axs[0].set_ylabel('Isa')
    axs[0].legend()
    axs[0].grid(True)

    # Wykres dla Isb (prąd)
    axs[1].plot(t, y[:, 1], label='Isb (prąd)', color='r')
    axs[1].set_xlabel('Czas (s)')
    axs[1].set_ylabel('Isb')
    axs[1].legend()
    axs[1].grid(True)

    # Wykres dla psi_ra (strumień)
    axs[2].plot(t, y[:, 2], label='psi_ra (strumień)', color='b')
    axs[2].set_xlabel('Czas (s)')
    axs[2].set_ylabel('psi_ra')
    axs[2].legend()
    axs[2].grid(True)

    # Wykres dla psi_rb (strumień)
    axs[3].plot(t, y[:, 3], label='psi_rb (strumień)', color='r')
    axs[3].set_xlabel('Czas (s)')
    axs[3].set_ylabel('psi_rb')
    axs[3].legend()
    axs[3].grid(True)

    # Wykres dla psi_ra (strumień)
    axs[4].plot(t, y_2[:, 0], label='Moduł prądu', color='b')
    axs[4].set_xlabel('Czas (s)')
    axs[4].set_ylabel('Is')
    axs[4].legend()
    axs[4].grid(True)

    # Wykres dla psi_rb (strumień)
    axs[5].plot(t, y_2[:, 1], label='Moduł strumienia', color='r')
    axs[5].set_xlabel('Czas (s)')
    axs[5].set_ylabel('Psi_r')
    axs[5].legend()
    axs[5].grid(True)
    
    plt.tight_layout()  # Automatyczne rozmieszczenie wykresów
    plt.show()

def main():
    # Pobieranie parametrów z GUI
    Ls =  float(entry_Ls.get())   # Indukcyjność stojana
    Lr =  float(entry_Lr.get())   # Indukcyjność wirnika
    Lm =  float(entry_Lm.get())   # Indukcyjność wzajemna
    t_end = int(entry_t.get())    # Czas przebiegu
    Rs = float(entry_Rs.get())    # Rezystancja stojana
    Rr = float(entry_Rr.get())    # Rezystancja wirnika
    J = float(entry_J.get())      # Moment bezwładności wirnika
    omega_r = float(entry_omega_r.get())   # Prędkość kątowa wirnika
    u_max = float(entry_u_s.get())   # Napięcie stojana (maksymalna wartość)
    h = float(entry_h.get())       # Krok symulacji
    f = 50  # Częstotliwość (50 Hz)

    # Warunki początkowe
    y0 = [0.1, 0.1, 0, 0, 0]  # [Isa, Isb, psi_ra, psi_rb, domega_r_dt]

    # Obliczenie współczynników
    w = Ls * Lr - Lm**2
    a1 = -(Rs * Lr**2 + Rr * Lm**2) / (w * Lr)
    a2 = (Rr * Lm) / (w * Lr)
    a3 = -(Lm / w)
    a4 = Lr / w
    a5 = (Rr * Lm) / Lr
    a6 = -Rr / Lr
    a7 = Lm/Lr

    # Wywołanie funkcji RK4
    t, y = runge_kutta(rownania_maszyny_induk, y0, t_end, h, a1, a2, a3, a4, a5, a6, a7, omega_r, u_max, f, J)
    t, y_mod = mod(t , y)
    y_2 = np.column_stack((y, y_mod))

    # Rysowanie wyników
    toDo(t, y, y_2)
# Tworzenie okna aplikacji
root = tk.Tk()
root.title("Symulacja silnika")

# Tworzenie pól do wprowadzania parametrów
tk.Label(root, text="Ls").pack()
entry_Ls = tk.Entry(root)
entry_Ls.pack()
entry_Ls.insert(0, "0.149")

tk.Label(root, text="Lr").pack()
entry_Lr = tk.Entry(root)
entry_Lr.pack()
entry_Lr.insert(0, "0.149")

tk.Label(root, text="Lm").pack()
entry_Lm = tk.Entry(root)
entry_Lm.pack()
entry_Lm.insert(0, "0.143")

tk.Label(root, text="Czas").pack()
entry_t = tk.Entry(root)
entry_t.pack()
entry_t.insert(0, "5")

tk.Label(root, text="Rs").pack()
entry_Rs = tk.Entry(root)
entry_Rs.pack()
entry_Rs.insert(0, "1.02")

tk.Label(root, text="Rr").pack()
entry_Rr = tk.Entry(root)
entry_Rr.pack()
entry_Rr.insert(0, "0.55")

tk.Label(root, text="Napięcie").pack()
entry_u_s = tk.Entry(root)
entry_u_s.pack()
entry_u_s.insert(0, "400")

tk.Label(root, text="Moment bezwładności").pack()
entry_J = tk.Entry(root)
entry_J.pack()
entry_J.insert(0, "0.2")

tk.Label(root, text="Prędkość kątowa").pack()
entry_omega_r = tk.Entry(root)
entry_omega_r.pack()
entry_omega_r.insert(0, "20")

tk.Label(root, text="Krok").pack()
entry_h = tk.Entry(root)
entry_h.pack()
entry_h.insert(0, "0.001")

# Tworzenie przycisku do uruchomienia symulacji
button = tk.Button(root, text="Uruchom symulację", command=main)
button.pack()

# Uruchomienie głównej pętli aplikacji
root.mainloop()
