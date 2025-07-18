distance = 16

# reading csv
data = pands.read_csv(r"15cm3.csv")

# Adjusts Digital signal to Analog and subtracts the base trim voltage to align 'base' value to 0 (based on principle that the average of the signal is 0)
analog = data * 3.3 / 65535
adj = np.mean(analog)
adjusted = analog - adj

# Gets array from first column of data
Audio = adjusted.iloc[:, 0]

# Adjusts the raw row numbers into seconds by dividing by the sampling rate
Duration = len(data)
Plots = np.arange(0, Duration, 1)
Time = Plots / 5000

# Finds Time values for where Audio Signal Values first become signifigant
xmin = np.argmax(Audio > 0.5)

# From first signifigant signal an arbitrary additional sample of 1000 is taken (roughly 0.2 seconds)
xmax = xmin + 1000

# finds points nearest to where signal crosses zero
root = np.where(np.diff(np.sign(Audio[xmin:xmax])))[0]

# Finds a more precise x position of the zero crossing.
Scatterx = []
for x in root:
    i = x + xmin
    y1, y2 = Audio[i], Audio[i + 1]
    x1, x2 = Time[i], Time[i + 1]
    m = (y2 - y1) / (x2 - x1)
    c1 = y1 - m * x1
    c2 = y2 - m * x2
    c = (c1 + c2) / 2
    Scatterx.append(-c / m)

# Filters out roots closer together than 500Hz
filtered_Scatterx = []
for i in range(len(Scatterx) - 1):
    t1, t2 = (
        Scatterx[i],
        Scatterx[i + 1],
    )
    freq = 1 / ((t2 - t1) / 2)
    if freq <= 500:
        filtered_Scatterx.append(t1)
        filtered_Scatterx.append(t2)


# Remove duplicates from filtered_Scatterx
filtered_Scatterx = list(dict.fromkeys(filtered_Scatterx))

# Calculates frequency based on the average times between zero crossings
frequency = (1 / np.mean(np.diff(filtered_Scatterx))) / 2
frequencyv = round(frequency, 2)
print(
    "Frequency calculated based on signal provided for a",
    distance,
    "cm overhang is",
    frequencyv,
    "Hz",
)

# Calculates the Youngs Modulus of the material based on the frequency and the length of the material
d = distance / 100
lengthm = 0.33
mass = 0.051
qkgpm = mass / lengthm
height = 0.00076
width = 0.0255
second = (width * height**3) / 12

E = (((frequency / 0.56) ** 2) * qkgpm * d**4) / second
Ev = round((E / 10**9), 2)
print(
    "Youngs Modulus calculated based on signal provided for a",
    distance,
    "cm overhang is",
    Ev,
    "GPa",
)


# plots the signal against time.
plt.scatter(
    filtered_Scatterx,
    [0] * len(filtered_Scatterx),
    color="blue",
    zorder=5,
    linewidths=1,
    marker="x",
)
plt.plot(Time, Audio, color="red", linestyle="solid", linewidth=0.5)
plt.axhline(y=0, color="black", linestyle="solid", linewidth=1)
ax = plt.gca()
ax.set_xlim(Time[xmin], Time[xmax])
ax.set_ylim(-adj - 0.1, adj + 0.1)
plt.grid(True, linestyle="solid", linewidth=0.45, which="major")
plt.grid(True, linestyle="solid", linewidth=0.15, which="minor")
plt.minorticks_on()
plt.tick_params(which="minor", length=3, color="black", width=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Audio Signal (V)")
plt.title("Signal vs Time")
plt.show()
