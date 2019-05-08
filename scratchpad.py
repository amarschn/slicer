import matplotlib.pyplot as plt
import pyclipper
import ipdb


polygon1 = ((0,0), (10,0), (10,10), (0,10))
polygon2 = ((20, 20), (30,20), (30,30), (20,30))
polygon3 = ((0,0),(3,0),(3,3),(0,3))
polygon4 = ((0,0),(3,0),(3,3),(0,3))
polygon5 = ((0,0),(3,0),(3,3),(0,3))
polygon6 = ((0,0),(3,0),(3,3),(0,3))
# pc = pyclipper.Pyclipper()
# # ipdb.set_trace()
# pc.AddPath(polygon1, pyclipper.PT_CLIP, True)
# pc.AddPath(polygon2, pyclipper.PT_CLIP, True)

# solution = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)

# print(tuple(map(tuple, solution[0])))

# line = ((0,0), (10,0), (10,10), (0,10), (0,0))
# pco = pyclipper.PyclipperOffset()
# pco.AddPath(line, pyclipper.JT_ROUND, pyclipper.ET_OPENROUND)
# solution = pco.Execute(2)
# print(solution)
# plt.plot(*zip(*solution[0]))
# plt.show()

H1 = ((-45.0, 50.0), (-45.0, 30.0), (-45.0, -50.0), (-39.0, -50.0), (-15.0, -50.0), (-15.0, -43.0), (-15.0, -15.0), (-9.0, -15.0), (15.0, -15.0), (15.0, -22.0), (15.0, -50.0), (21.0, -50.0), (45.0, -50.0), (45.0, -30.0), (45.0, 50.0), (39.0, 50.0), (15.0, 50.0), (15.0, 43.0), (15.0, 15.0), (9.0, 15.0), (-15.0, 15.0), (-15.0, 22.0), (-15.0, 50.0), (-21.0, 50.0))
H2 = ((-45.0, 50.0), (-45.0, 30.0), (-45.0, -50.0), (-39.0, -50.0), (-15.0, -50.0), (-15.0, -43.0), (-15.0, -15.0), (-9.0, -15.0), (15.0, -15.0), (15.0, -22.0), (15.0, -50.0), (21.0, -50.0), (45.0, -50.0), (45.0, -30.0), (45.0, 50.0), (39.0, 50.0), (15.0, 50.0), (15.0, 43.0), (15.0, 15.0), (9.0, 15.0), (-15.0, 15.0), (-15.0, 22.0), (-15.0, 50.0), (-21.0, 50.0))
pc = pyclipper.Pyclipper()
# ipdb.set_trace()
# pc.AddPath(H1, pyclipper.PT_CLIP, True)
# pc.AddPath(H2, pyclipper.PT_CLIP, True)
pc.AddPath(polygon3, pyclipper.PT_CLIP, True)
pc.AddPath(polygon4, pyclipper.PT_CLIP, True)
pc.AddPath(polygon5, pyclipper.PT_CLIP, True)
pc.AddPath(polygon6, pyclipper.PT_SUBJECT, True)

solution = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)

if solution:
	print(solution)
	plt.plot(*zip(*solution[0]), lineStyle='None', marker='o')
	plt.show()
else:
	print("No solution")