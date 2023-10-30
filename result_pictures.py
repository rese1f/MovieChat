import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

x = [1, 2, 3, 4, 5]
y11 = [65.3, 67.8, 61.6, 59.8, 58.5]
y12 = [65.9, 65.7, 65.3, 63.7, 63.3]

y21 = [57.5, 59.3, 59.5, 61.5, 61.8]



color_y2 = '#7467B5'  
color_y2_point = '#BEB8DC' 
color_y1 = '#3F938B'  
color_y1_point = '#8ECFC9'  
color_y3 = '#33668D'  
color_y3_point = '#82B0D2' 


fig, ax1 = plt.subplots(figsize=(10,5))


line1 = ax1.plot(x, y12, marker='o', color=color_y1, linestyle='--', markerfacecolor=color_y1_point, linewidth=6, markersize=12, label='$l_{long}=2^{6}$')

ax1.set_xlabel('Length of merged tokens', fontsize=28)

ax1.set_ylabel('Accuracy', color='#000000', fontsize=28)
ax1.tick_params('y', colors='#000000', labelsize=26)
ax1.tick_params(axis='x', labelsize=26)

# ax2 = ax1.twinx()
line2 = ax1.plot(x, y11, marker='o', color=color_y2, markerfacecolor=color_y2_point, linewidth=6, markersize=12, label='$l_{long}=2^{8}$')


# ax3 = ax1.twinx()
line3 = ax1.plot(x, y21, marker='o', color=color_y3, linestyle='--', markerfacecolor=color_y3_point, linewidth=6, markersize=12, label='$l_{long}=2^{9}$')

lines = line1 + line2 + line3 
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best', fontsize=24)

ax1.set_xticks(x)
ax1.set_xticklabels([f'{i}' for i in x])

# plt.xscale('log', base=2)

plt.tight_layout()

# plt.savefig('double_line_chart.jpg', format='jpg')
plt.savefig('example_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# # x = [pow(2,6), pow(2,7), pow(2,8), pow(2,9), pow(2,10)]
# # y11 = [65.9, 66.3, 67.8, 61.8, 58.5]
# # y12 = [3.66, 3.77, 3.81, 3.67, 3.42]

# x = [8, 12, 16, 20, 24]
# y11 = [65.7, 67.2, 67.8, 66.5, 63.6]
# y12 = [3.66, 3.72, 3.81, 3.69, 3.32]


# color_y1 = '#7467B5'  
# color_y1_point = '#BEB8DC' 
# color_y2 = '#C8AA98'  
# color_y2_point = '#E7DAD2' 
 
# fig, ax1 = plt.subplots(figsize=(10,5))

# line1 = ax1.plot(x, y11, marker='o', color=color_y1, markerfacecolor=color_y1_point, linewidth=6, markersize=12, label='Accuracy')

# ax1.set_xlabel('Length of short-term memory buffer', fontsize=28)

# ax1.set_ylabel('Accuracy', color='#000000', fontsize=28)
# ax1.tick_params('y', colors='#000000', labelsize=26)
# ax1.tick_params(axis='x', labelsize=26)


# ax2 = ax1.twinx()
# line2 = ax2.plot(x, y12, marker='o', color=color_y2, markerfacecolor=color_y2_point, linewidth=6, markersize=12, label='Score')

# ax2.set_ylabel('Score', color='#000000', fontsize=28)
# ax2.tick_params('y', colors='#000000', labelsize=26)
# ax2.tick_params(axis='x', labelsize=26)

# lines = line1 + line2 
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='lower left', fontsize=24)

# ax1.set_xticks(x)
# ax1.set_xticklabels([f'{i}' for i in x])

# # plt.xscale('log', base=2)
# plt.tight_layout()

# # plt.savefig('double_line_chart.jpg', format='jpg')
# plt.savefig('example_plot.pdf', dpi=300, bbox_inches='tight')
# plt.show()