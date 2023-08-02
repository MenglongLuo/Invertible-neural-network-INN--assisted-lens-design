import torch
import numpy as np
import model
from torch.autograd import Variable
import configuration as c
import data as d
import win32com.client
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

filename = d.pklname
pretrained_model = torch.load(filename, map_location = lambda storage, loc: storage)
generator = model.model
generator.load_state_dict(pretrained_model)

### Set targets of the vertical divergence angle and horizontal deflection and divergence angles ###
target_ver_div = 4.22
target_steer = 50.2
target_hor_div = target_steer / 15 * 0.23

### 0 = lens V and 1 = lens H
start_time = time.time()

if d.lens in [0, 1, 2, 3]:
    y_target = [target_ver_div]
elif d.lens in [4, 5]:
    y_target = [target_steer, target_hor_div]
    
y_transform = d.scaler_y.transform([y_target])

## Number of the generated samples. Here we set a large number and average the samples to make the result more scientific.

n_samps = 10000
y_fix = np.zeros((n_samps, len(y_target))) + y_transform
y_fix = torch.tensor(y_fix, dtype = torch.float)

# y_fix += c.add_y_noise * torch.randn(n_samps, c.ndim_y)
y_fix = torch.cat([torch.randn(n_samps, c.ndim_z), c.add_z_noise * torch.zeros(n_samps, 0), y_fix], dim=1)
y_fix = y_fix.to(device)

# posterior samples
rev_x0 = generator(y_fix, rev=True)[0]

### Utilize the prediceted X to predict Y ###
# out_y0 = generator(rev_x0)[0]
# print(out_y0[:,-1])

## Save the predicted X ###
rev_x = rev_x0.cpu().data.numpy()
rev_x = torch.tensor(d.inverse_transform_x(rev_x))
rev_x = torch.mean(rev_x, dim=0)
rev_x = np.array(rev_x.detach().cpu())
# rev_y = out_y0.cpu().data.numpy()
fname = 'gen_samps_inn_x.csv'
# fname = 'gen_samps_inn_y.csv'
np.savetxt(fname, rev_x, fmt='%.5f', delimiter=',')
# np.savetxt(fname, rev_y, fmt='%.2f', delimiter=',')

end_time = time.time()

elapsed_time = end_time - start_time

print("Elapsed time：", elapsed_time, ' s')
### Calculate the absolute percentage error ###
def absolute_percentage_error(target, prediction):
    return abs((target - prediction) / target) * 100

### LightTools ###

# Create the LightTools application object
lt = win32com.client.Dispatch('LightTools.LTAPI4')

# # Set the parameters of the lens and the receiver, then run the simulation and check the values of the receiver.
if d.lens in [0, 1, 2, 3]:
    # Set the parameters of the lens V
    LensV = 'V_1'
    LensV_WorkingDistance = f'SOLID[{LensV}]'
    LensV_Thickness = f'SOLID[{LensV}].RECT_LENS_PRIMITIVE[LP_1]'
    LensV_RearSurface = f'SOLID[{LensV}].RECT_LENS_PRIMITIVE[LP_1].YTOROID_LENS_SURFACE[LensRearSurface]'
    Receiver = 'RECEIVERS[Receiver_List].FARFIELD_RECEIVER[farFieldReceiver_91].FORWARD_SIM_FUNCTION[Forward_Simulation].INTENSITY_MESH[Intensity_Mesh]'
    WorkingDistance = lt.DbSet(LensV_WorkingDistance, 'Z', rev_x[0])
    Lens_Thickness = lt.DbSet(LensV_Thickness, 'Center_Thickness', rev_x[1])
    Rear_Conic_Constant = lt.DbSet(LensV_RearSurface, 'Conic_Constant', rev_x[2])
    Rear_Radius = lt.DbSet(LensV_RearSurface, 'Radius', rev_x[3])
    
    # Run the LightTools simulation
    lt.CMD('StartSim')
    
    # Get the results of the lens V
    VerticalDivergenceAngle = lt.DbGet(Receiver, 'Full_Width_1', 3, 3)[0]
    APE_VerticalAngle = absolute_percentage_error(target_ver_div, VerticalDivergenceAngle)
    
    print('Target vertical divergence angle: {:.2f}°'.format(y_target[0]))
    print('Predicted vertical divergence angle: {:.2f}°'.format(VerticalDivergenceAngle))
    print("Absolute percentage error：{:.2f}%".format(APE_VerticalAngle))
    print('WD, Thickness, CC, Radius:\n', np.round(rev_x, 2))

elif d.lens in [4, 5]:
    # Set the parameters of the lens H
    LensH = 'H1'
    LensH_WorkingDistance = f'SOLID[{LensH}]'
    LensH_Thickness = f'SOLID[{LensH}].RECT_LENS_PRIMITIVE[LP_1]'
    LensH_FrontSurface = f'SOLID[{LensH}].RECT_LENS_PRIMITIVE[LP_1].YTOROID_LENS_SURFACE[LensFrontSurface]'
    LensH_RearSurface = f'SOLID[{LensH}].RECT_LENS_PRIMITIVE[LP_1].YTOROID_LENS_SURFACE[LensRearSurface]'
    Receiver = 'RECEIVERS[Receiver_List].FARFIELD_RECEIVER[farFieldReceiver_91].FORWARD_SIM_FUNCTION[Forward_Simulation].INTENSITY_MESH[Intensity_Mesh]'
    WorkingDistance = lt.DbSet(LensH_WorkingDistance, 'Z', rev_x[0])
    Thickness = lt.DbSet(LensH_Thickness, 'Center_Thickness', rev_x[1])
#     Front_Conic_Constant = lt.DbSet(LensH_FrontSurface, 'Conic_Constant', rev_x[2])
    Front_Radius= lt.DbSet(LensH_FrontSurface, 'Radius', rev_x[2])
    Rear_Radius = lt.DbSet(LensH_RearSurface, 'Radius', rev_x[3])
    
    # Run the LightTools simulation
    lt.CMD('StartSim')
    
    # Get the results of the lens H
    predict_steer = lt.DbGet(Receiver, 'Beam_Peak_X')[0]
    predict_hor_div = lt.DbGet(Receiver, 'Full_Width_1', 3, 2)[0]
    APE_SteerAngle = absolute_percentage_error(target_steer, predict_steer)
    APE_HorizontalDivergenceAngle = absolute_percentage_error(target_hor_div, predict_hor_div)
    y_prediction = np.array([predict_steer, predict_hor_div])
    
    print('Target steering and hor_divergence:', y_target)
    print('Predicted steering and hor_divergence:', y_prediction)
    print("APE_SteerAngle：{:.2f}%".format(APE_SteerAngle))
    print("APE_HorizontalDivergenceAngle：{:.2f}%".format(APE_HorizontalDivergenceAngle))
    print('WD, Thickness, FR, RR:\n', np.round(rev_x, 2))
