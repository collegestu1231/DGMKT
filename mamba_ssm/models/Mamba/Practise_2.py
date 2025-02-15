import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define the data
# data = [[0.2356, 0.1582, 0.3354, 0.3385, 0.3148, 0.2986, 0.3188, 0.3436, 0.3494, 0.3450, 0.4169, 0.5153, 0.5406, 0.3563, 0.5615, 0.5392, 0.4146, 0.3518, 0.2964, 0.2417, 0.2500, 0.1980, 0.3149, 0.3165, 0.3728, 0.2704, 0.2827, 0.2944, 0.3737, 0.2373, 0.1936, 0.3294, 0.2778, 0.3327, 0.2621, 0.2917],
#  [0.4903, 0.4041, 0.3365, 0.3117, 0.3014, 0.2801, 0.2615, 0.2424, 0.2241, 0.2104, 0.4231, 0.6053, 0.7203, 0.3187, 0.5116, 0.3468, 0.1358, 0.2044, 0.2099, 0.2430, 0.4234, 0.5663, 0.5740, 0.1989, 0.0893, 0.1369, 0.1963, 0.1126, 0.0583, 0.1206, 0.3759, 0.2864, 0.4761, 0.6540, 0.6022, 0.6226],
#  [0.5278, 0.4799, 0.4691, 0.3973, 0.3849, 0.3704, 0.3546, 0.3398, 0.3278, 0.3216, 0.1776, 0.1778, 0.2308, 0.3902, 0.4549, 0.4563, 0.4236, 0.4442, 0.1818, 0.4944, 0.5952, 0.5696, 0.2247, 0.2433, 0.3912, 0.2968, 0.1082, 0.2889, 0.4361, 0.1936, 0.2232, 0.2621, 0.2885, 0.3103, 0.4345, 0.4918],
#  [0.5043, 0.4738, 0.4293, 0.3240, 0.3365, 0.3510, 0.3506, 0.3387, 0.3291, 0.3235, 0.2184, 0.2166, 0.2219, 0.2859, 0.2235, 0.4794, 0.4937, 0.1772, 0.4983, 0.2521, 0.2561, 0.3111, 0.5534, 0.5266, 0.4568, 0.1601, 0.4916, 0.5664, 0.5304, 0.1896, 0.3912, 0.2279, 0.2087, 0.2045, 0.2569, 0.2824],
#  [0.5225, 0.4558, 0.4363, 0.4014, 0.4109, 0.3995, 0.3788, 0.3558, 0.3374, 0.3257, 0.3211, 0.3791, 0.4367, 0.4094, 0.3414, 0.3685, 0.4687, 0.2727, 0.3492, 0.2800, 0.2545, 0.2924, 0.3683, 0.4682, 0.4428, 0.2705, 0.3349, 0.3857, 0.4235, 0.2474, 0.3732, 0.3016, 0.3809, 0.4648, 0.5340, 0.5444]]
# # DKT
# data = [
#      [0.2305, 0.2437, 0.3584, 0.3268, 0.3136, 0.3042, 0.2976, 0.294, 0.2912, 0.2886, 0.5269, 0.5386, 0.5052, 0.3437, 0.6227, 0.3358, 0.1842, 0.3519, 0.265, 0.3843, 0.3624, 0.4505, 0.3458, 0.2413, 0.2598, 0.3259, 0.2373, 0.2338, 0.3027, 0.304, 0.3849, 0.3562, 0.4268, 0.4043, 0.4124, 0.4205],
#      [0.3421, 0.2188, 0.2683, 0.2534, 0.2243, 0.2064, 0.1999, 0.1935, 0.1879, 0.1834, 0.3798, 0.4386, 0.4662, 0.3295, 0.5327, 0.3571, 0.2337, 0.4303, 0.2565, 0.3773, 0.4054, 0.4869, 0.3854, 0.2507, 0.2546, 0.4406, 0.1964, 0.2161, 0.2692, 0.3829, 0.4163, 0.2545, 0.4337, 0.4852, 0.5455, 0.5793],
#      [0.4283, 0.3277, 0.3835, 0.3491, 0.3041, 0.2844, 0.2736, 0.2643, 0.2562, 0.2493, 0.3424, 0.3358, 0.3071, 0.4119, 0.3198, 0.3746, 0.4375, 0.6195, 0.3345, 0.4461, 0.5409, 0.647, 0.3753, 0.3336, 0.3394, 0.3403, 0.2291, 0.2823, 0.3677, 0.2745, 0.3844, 0.3048, 0.397, 0.4018, 0.358, 0.4164],
#      [0.3966, 0.3627, 0.4418, 0.3658, 0.3313, 0.3093, 0.3025, 0.2963, 0.2913, 0.2868, 0.2521, 0.1892, 0.2214, 0.4445, 0.3329, 0.4093, 0.3479, 0.3138, 0.2848, 0.3404, 0.3042, 0.3545, 0.4073, 0.3555, 0.3524, 0.2289, 0.2169, 0.2441, 0.3678, 0.2376, 0.2879, 0.2472, 0.4043, 0.2693, 0.3096, 0.3229],
#      [0.4686, 0.4587, 0.4129, 0.4094, 0.4058, 0.3915, 0.379, 0.371, 0.3645, 0.3587, 0.3008, 0.3303, 0.3505, 0.3767, 0.3549, 0.3845, 0.3424, 0.3101, 0.3317, 0.3246, 0.3778, 0.4629, 0.443, 0.4016, 0.3848, 0.3321, 0.3267, 0.3434, 0.3791, 0.2957, 0.3876, 0.4154, 0.4176, 0.4781, 0.5657, 0.5385],]
# Mamba
data = [
    [0.2001, 0.2206, 0.3224, 0.3035, 0.2953, 0.2821, 0.2756, 0.2734, 0.2715, 0.2684, 0.5609, 0.5801, 0.5244, 0.3597, 0.6295, 0.3640, 0.2034, 0.3863, 0.2805, 0.4073, 0.3592, 0.4028, 0.3209, 0.2387, 0.2718, 0.2756, 0.2312, 0.2255, 0.3025, 0.2564, 0.3668, 0.3357, 0.4528, 0.3989, 0.3732, 0.3769],
    [0.4876, 0.4575, 0.3794, 0.3795, 0.3280, 0.2934, 0.2722, 0.2610, 0.2507, 0.2420, 0.4826, 0.5227, 0.5430, 0.3392, 0.5375, 0.5101, 0.3797, 0.3622, 0.3415, 0.3845, 0.4305, 0.4704, 0.4095, 0.3114, 0.3462, 0.5166, 0.2516, 0.2263, 0.3964, 0.4843, 0.3335, 0.2994, 0.4653, 0.5471, 0.6059, 0.6255],
    [0.4978, 0.3532, 0.3478, 0.3183, 0.2974, 0.2887, 0.2750, 0.2689, 0.2626, 0.2563, 0.2909, 0.2678, 0.2860, 0.3624, 0.3208, 0.4065, 0.3992, 0.6075, 0.3167, 0.4534, 0.5062, 0.5932, 0.3715, 0.3166, 0.3241, 0.3141, 0.2154, 0.2672, 0.3559, 0.2484, 0.3808, 0.3028, 0.4444, 0.3910, 0.3662, 0.4047],
    [0.4263, 0.3494, 0.4233, 0.3463, 0.3330, 0.3231, 0.3200, 0.3160, 0.3117, 0.3074, 0.2335, 0.1758, 0.2103, 0.3964, 0.3095, 0.3796, 0.3004, 0.3766, 0.2512, 0.3594, 0.2868, 0.3252, 0.4100, 0.3479, 0.3857, 0.2202, 0.2006, 0.2373, 0.3873, 0.2186, 0.3973, 0.2454, 0.4672, 0.3002, 0.2709, 0.2899],
    [0.4407, 0.4261, 0.3691, 0.3759, 0.3751, 0.3687, 0.3516, 0.3450, 0.3395, 0.3340, 0.2651, 0.3089, 0.3345, 0.3281, 0.3143, 0.3686, 0.3362, 0.2906, 0.3155, 0.3089, 0.3503, 0.4345, 0.4059, 0.3668, 0.3504, 0.3131, 0.3105, 0.3114, 0.3625, 0.2823, 0.3161, 0.3630, 0.3769, 0.4390, 0.5323, 0.5138]
]

# Row and column labels
row_labels = [r'$C_{54}$', r'$C_{74}$', r'$C_{14}$', r'$C_{69}$', r'$C_{21}$']
col_labels = [str(i) for i in range(1, 37)]

# Create a DataFrame
df = pd.DataFrame(data, index=row_labels, columns=col_labels)
vmin = np.min(data)
vmax = np.max(data)
# Initialize the figure
fig, ax = plt.subplots(figsize=(12, 6))


# Create an axes divider for precise colorbar placement
# divider = make_axes_locatable(ax)
#
# # Add a colorbar axis
# cax = divider.append_axes("right", size="5%", pad=0.1)
# Draw the heatmap
# sns.heatmap(df, annot=False, cmap='RdYlGn', linewidths=0.5, linecolor='black', square=True,
#             cbar_ax=cax, vmin=vmin, vmax=vmax, ax=ax)
sns.heatmap(df, annot=False, cmap='Blues', linewidths=0.5, square=True,
            cbar_kws={"shrink": 1.0, "aspect": 30,"pad": 0.02}, vmin=vmin, vmax=vmax, ax=ax)
# sns.heatmap(df, annot=False, cmap='RdYlGn',linewidths=0.5, square=True, cbar_kws={"shrink": 0.8, "aspect": 30}, ax=ax)
# sns.heatmap(df, annot=False, cmap='RdYlGn', linewidths=0.5, linecolor='black', square=True, cbar_kws={"shrink": 0.8, "aspect": 20}, ax=ax)

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# # Add top circles for col_labels
# top_colors = ['#006A67', '#0072b2', '#003161', '#BAD1C2', '#041C32'] * 7 + ['#006A67']  # Repeat colors as needed
# for i, color in enumerate(top_colors):
#     circle = plt.Circle((i + 0.5, -0.5), 0.4, edgecolor=color, facecolor='none', linewidth=2, clip_on=False)
#     ax.add_artist(circle)
# Add top circles for col_labels
# top_colors = ['#006A67', '#0072b2', '#003161', '#BAD1C2', '#041C32'] * 7 + ['#006A67']  # Repeat colors as needed
# for i, color in enumerate(top_colors):
#     # Use facecolor=color to make the circles solid
#     circle = plt.Circle((i + 0.5, -0.5), 0.4, edgecolor=color, facecolor=color, linewidth=2, clip_on=False)
#     ax.add_artist(circle)

# Define top circle colors and hollow/solid status
# Define top circle colors and hollow/solid status
top_circle_specs = [
    ('#006A67', True),   # 1-10 红色空心
    ('#006A67', True),
    ('#006A67', True),
    ('#006A67', True),
    ('#006A67', True),
    ('#006A67', True),
    ('#006A67', True),
    ('#006A67', True),
    ('#006A67', True),
    ('#006A67', True),
    ('#006A67', False),  # 11-13 红色实心
    ('#006A67', False),
    ('#006A67', False),
    ('#0072b2', True),   # 14 蓝色空心
    ('#0072b2', False),  # 15 蓝色实心
    ('#003161', True),   # 16-17 紫色空心
    ('#003161', True),
    ('#003161', False),  # 18 紫色实心
    ('#003161', True),   # 19 紫色空心
    ('#003161', False),  # 20-22 紫色实心
    ('#003161', False),
    ('#003161', False),
    ('#003161', True),   # 23-25 紫色空心
    ('#003161', True),
    ('#003161', True),
    ('#003161', False),  # 26 紫色实心
    ('#003161', True),   # 27-29 紫色空心
    ('#003161', True),
    ('#003161', True),
    ('#003161', False),  # 30 紫色实心
    ('#BAD1C2', False),  # 31 橙色实心
    ('#041C32', True),   # 32 蓝色空心
    ('#041C32', False),  # 33-36 蓝色实心
    ('#041C32', False),
    ('#041C32', False),
    ('#041C32', False),
]

# Add top circles for col_labels
for i, (color, hollow) in enumerate(top_circle_specs):
    if color != 'none':  # Ensure a valid color is specified
        if hollow:
            # If hollow, set facecolor='none'
            circle = plt.Circle((i + 0.5, -0.5), 0.4, edgecolor=color, facecolor='none', linewidth=2, clip_on=False)
        else:
            # If not hollow, set facecolor=color
            circle = plt.Circle((i + 0.5, -0.5), 0.4, edgecolor=color, facecolor=color, linewidth=2, clip_on=False)
        ax.add_artist(circle)


# Add left circles for row_labels
side_colors = ['#006A67', '#0072b2', '#003161', '#BAD1C2', '#041C32']
for j, color in enumerate(side_colors):
    circle = plt.Circle((-0.5, j + 0.5), 0.4, color=color, fill=True, clip_on=False)
    ax.add_artist(circle)

# Adjust the plot
plt.title("● correct     ○ incorrect ", fontsize=20, loc='center')
# plt.xlabel("Column Labels")
# plt.ylabel("Row Labels")
plt.xticks(ticks=np.arange(0.5, len(col_labels), 1), labels=col_labels, rotation=45, ha='right', fontsize=10)
plt.yticks(ticks=np.arange(0.5, len(row_labels), 1), labels=row_labels, rotation=0, fontsize=10)

# Ensure proper limits to show circles
ax.set_xlim(-1, len(col_labels))
ax.set_ylim(len(row_labels), -1)

# Adjust colorbar to match heatmap size
cbar = ax.collections[0].colorbar
cbar.set_ticks(np.linspace(vmin, vmax, 5))
cbar.ax.tick_params(labelsize=10)
cbar.ax.set_aspect(15)
# cbar.ax.set_aspect(20)

# Tight layout for better spacing
plt.tight_layout()
plt.savefig("heatmap_output.pdf", format="pdf", bbox_inches="tight")
# Display the plot
plt.show()
