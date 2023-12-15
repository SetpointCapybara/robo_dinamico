from tqdm import tqdm
import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt

def desired_velocity(robot, q, t, r2=False):
  target = np.array([[-6.34135828098418e-07, 0.9999996824268551, -0.0007969603419838722, 0.0014655383006834171],
                     [0.9999996834368142, 4.0212912450946496e-13, -0.00079569232221419, 0.5607591952042734],
                     [-0.0007956920695233563, -0.000796960594272577, -0.9999993658637698, 0.3],
                     [0.0, 0.0, 0.0, 1.0]])
  
  des_p = target[:3, -1].reshape((3, 1))
  des_x = target[:3, 0].reshape((3, 1))
  des_y = target[:3, 1].reshape((3, 1))
  des_z = target[:3, 2].reshape((3, 1))

  jac_geo, htm_out = robot.jac_geo(q)

  p_eef = htm_out[:3, -1].reshape((3, 1))
  x_eef = htm_out[:3, 0].reshape((3, 1))
  y_eef = htm_out[:3, 1].reshape((3, 1))
  z_eef = htm_out[:3, 2].reshape((3, 1))

  jac_pos = jac_geo[:3, :]
  jac_ori = jac_geo[3:, :]

  r = np.vstack((p_eef-des_p, 1 - des_x.T @ x_eef, 1 - des_y.T @ y_eef, 1 - des_z.T @ z_eef))
  
  Jr = np.vstack((jac_pos,des_x.T@ub.Utils.S(x_eef)@jac_ori, des_y.T@ub.Utils.S(y_eef)@jac_ori, des_z.T@ub.Utils.S(z_eef)@jac_ori))

  k = 20

  u = ub.Utils.dp_inv(Jr) @ (-k*r)

  u = np.clip(u, -0.5, 0.5)

  if r2==True:
    return u, r
  
  return u

def desired_velocity2(robot, q, t, r2=False):
  target = np.array([[ 7.96960342e-04,  9.99999682e-01, -6.34135828e-07,  1.06705813e-03],
                     [ 7.95692322e-04,  4.02129125e-13,  9.99999683e-01,  5.60361349e-01],
                     [ 9.99999366e-01, -7.96960594e-04, -7.95692070e-04,  7.89613287e-01],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
  
  des_p = target[:3, -1].reshape((3, 1))
  des_x = target[:3, 0].reshape((3, 1))
  des_y = target[:3, 1].reshape((3, 1))
  des_z = target[:3, 2].reshape((3, 1))

  jac_geo, htm_out = robot.jac_geo(q)

  p_eef = htm_out[:3, -1].reshape((3, 1))
  x_eef = htm_out[:3, 0].reshape((3, 1))
  y_eef = htm_out[:3, 1].reshape((3, 1))
  z_eef = htm_out[:3, 2].reshape((3, 1))

  jac_pos = jac_geo[:3, :]
  jac_ori = jac_geo[3:, :]

  r = np.vstack((p_eef-des_p, 1 - des_x.T @ x_eef, 1 - des_y.T @ y_eef, 1 - des_z.T @ z_eef))
  
  Jr = np.vstack((jac_pos,des_x.T@ub.Utils.S(x_eef)@jac_ori, des_y.T@ub.Utils.S(y_eef)@jac_ori, des_z.T@ub.Utils.S(z_eef)@jac_ori))

  k = 20

  u = ub.Utils.dp_inv(Jr) @ (-k*r)

  u = np.clip(u, -0.5, 0.5)

  if r2==True:
    return u, r
  
  return u

#Cria robo
robot = ub.Robot.create_kuka_kr5()
real_robot = ub.Robot.create_kuka_kr5_per()

#Cria caixa
cube_position = ub.Utils.trn([0, 0.5, 0.15]) 
cube = ub.Box(htm=cube_position, width=0.3, depth=0.3, height=0.3, color="brown") 

#Coloca no cenario
sim = ub.Simulation([cube, robot])

#declara o dt
dt = 5e-4
#Outras declaracoes
q = robot.q
z = np.zeros((6, 1))
b = 5
kp = 40
ki = (kp**2)/4
i = 0

r_history1 = np.array([])
r_history2 = np.array([])
r_history3 = np.array([])
r_history4 = np.array([])
r_history5 = np.array([])
r_history6 = np.array([])

taum_history1 = np.array([])
taum_history2 = np.array([])
taum_history3 = np.array([])
taum_history4 = np.array([])
taum_history5 = np.array([])
taum_history6 = np.array([])

#Laco principal
for k in tqdm(range(np.ceil(5/dt).astype(np.int64))):
  #Calcula o tempo
  t = k*dt

  #Calcula os tres elementos
  dyn_m, dyn_c, dyn_g = robot.dyn_model(q, z)
  real_dyn_m, real_dyn_c, real_dyn_g = real_robot.dyn_model(q, z)

  #Calcula o torque do motor (taum)
  u, r = desired_velocity(robot, q, t, r2=True)
  r_history1 = np.append(r_history1, r[0])
  r_history2 = np.append(r_history2, r[1])
  r_history3 = np.append(r_history3, r[2])
  r_history4 = np.append(r_history4, r[3])
  r_history5 = np.append(r_history5, r[4])
  r_history6 = np.append(r_history6, r[5])

  dot_u = (desired_velocity(robot, q + z*dt, t + dt) - desired_velocity(robot, q - z*dt, t - dt))/2*dt
  delta_v = z - u
  i = i + delta_v*dt
  a = dot_u - kp*delta_v - ki*i
  taum = dyn_m @ a + dyn_c + dyn_g
  taum_history1 = np.append(taum_history1, taum[0])
  taum_history2 = np.append(taum_history2, taum[1])
  taum_history3 = np.append(taum_history3, taum[2])
  taum_history4 = np.append(taum_history4, taum[3])
  taum_history5 = np.append(taum_history5, taum[4])
  taum_history6 = np.append(taum_history6, taum[5])


  #Torque externo generalizado
  tau = taum

  #Calcula o "a2"
  a2 = np.linalg.inv(real_dyn_m)*(tau - real_dyn_c - real_dyn_g)

  #integra as equacoes
  q = q + z*dt
  z = z + a2*dt

  #coloca a configuracao no robo
  robot.add_ani_frame(k*dt, q = q)

robot.attach_object(cube)

for k in tqdm(range(np.ceil(5/dt).astype(np.int64))):
  #Calcula o tempo
  t = k*dt

  #Calcula os tres elementos
  dyn_m, dyn_c, dyn_g = robot.dyn_model(q, z)
  real_dyn_m, real_dyn_c, real_dyn_g = real_robot.dyn_model(q, z)

  #calcula a Jacobiana de posicao para a particula que esta na ponta do robo
  j, htm_out = robot.jac_geo()
  j_h = j[:3, :]

  #Calcula o torque do motor (taum)
  u, r = desired_velocity2(robot, q, t, r2=True)
  r_history1 = np.append(r_history1, r[0])
  r_history2 = np.append(r_history2, r[1])
  r_history3 = np.append(r_history3, r[2])
  r_history4 = np.append(r_history4, r[3])
  r_history5 = np.append(r_history5, r[4])
  r_history6 = np.append(r_history6, r[5])

  dot_u = (desired_velocity2(robot, q + z*dt, t + dt) - desired_velocity2(robot, q - z*dt, t - dt))/2*dt
  delta_v = z - u
  i = i + delta_v*dt
  a = dot_u - kp*delta_v - ki*i
  taum = dyn_m @ a + dyn_c + dyn_g

  taum_history1 = np.append(taum_history1, taum[0])
  taum_history2 = np.append(taum_history2, taum[1])
  taum_history3 = np.append(taum_history3, taum[2])
  taum_history4 = np.append(taum_history4, taum[3])
  taum_history5 = np.append(taum_history5, taum[4])
  taum_history6 = np.append(taum_history6, taum[5])

  #Torque externo generalizado
  tau = taum + j_h.T @ np.array([0, 0, -100]).reshape((3, 1))

  #Calcula o "a2"
  a2 = np.linalg.inv(real_dyn_m)*(tau - real_dyn_c - real_dyn_g)

  #integra as equacoes
  q = q + z*dt
  z = z + a2*dt

  #coloca a configuracao no robo
  robot.add_ani_frame(k*dt+5, q = q)

plt.plot(r_history1, label="r1")
plt.plot(r_history2, label="r2")
plt.plot(r_history3, label="r3")
plt.plot(r_history4, label="r4")
plt.plot(r_history5, label="r5")
plt.plot(r_history6, label="r6")
plt.title("Função de tarefa")
plt.xlabel("Passo de integração numérica")
plt.ylabel("Valor")
plt.legend()
plt.show()

plt.plot(taum_history1, label="t1")
plt.plot(taum_history2, label="t2")
plt.plot(taum_history3, label="t3")
plt.plot(taum_history4, label="t4")
plt.plot(taum_history5, label="t5")
plt.plot(taum_history6, label="t6")

plt.title("Torque enviado aos motores")
plt.xlabel("Passo de integração numérica")
plt.ylabel("Torque (Nm)")
plt.legend()
plt.show()

sim.save("/home/enacom/uaibot_coriolis/", "robo_dinamico")