clear 
close all
grey = [0.4 0.4 0.4];
addpath('..\00_utils')
set_plot_style

%%
load('kuramoto_sivashinsky.mat');

ux = zeros(size(usave));
uxx = zeros(size(usave));
uxxx = zeros(size(usave));
uxxxx = zeros(size(usave));
% compute derivatives
for t = 1:length(tsave)
    ux(t,2:end) = diff(usave(t,:))/dx;
    uxx(t,3:end) = diff(ux(t,2:end))/dx;
    uxxx(t,4:end) = diff(uxx(t,3:end))/dx;
    uxxxx(t,5:end) = diff(uxxx(t,4:end))/dx;
end    
% create input/output layer
x_train_index = 5:8:1024;
input = [];
output = [];
for jj = 1:length(x_train_index)
    x_pos = x_train_index(jj)*ones(length(tsave)-1,1);
    k = x_train_index(jj);
    input = [input; usave(1:end-1,k),ux(1:end-1,k),uxx(1:end-1,k),uxxxx(1:end-1,k),x_pos];
    output = [output; usave(2:end,k)];
end    


%% Training
TRAIN = 0;
if TRAIN
    net = feedforwardnet([30 30]);
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.transferFcn = 'radbas';
    net.trainParam.epochs = 2000;
    net.trainParam.max_fail = 1000;
    [net,tr] = train(net,input.',output.');
end


%% Load network trained
load('KS_NN2_30_grid8.mat')

%% Training Set: estimation performance
u_net = net(input.')';

for jj = 1:length(x_train_index)
    pp = jj-1;
    u_net_time(1:70,jj) = u_net(1+70*pp:70*jj);
    output_time(1:70,jj) = output(1+70*pp:70*jj);
end

figure; hold on; grid on
for kk = [20 50 80]
    plot(tsave(2:end),output_time(:,kk),'color',grey,'LineWidth',1.5,'DisplayName',"KS x="+num2str(x_train_index(kk)))
    plot(tsave(2:end),u_net_time(:,kk),'--','LineWidth',1.5,'DisplayName',"NN x="+num2str(x_train_index(kk)))
end
xlabel('time [s]')
ylabel("u")
legend show

for jj = 1:length(x_train_index)
    RMSE(jj) = rms(output_time(:,jj)-u_net_time(:,jj))/rms(output_time(:,jj))*100;
end
figure; 
plot(x_train_index,RMSE,'*','LineWidth',1.5)
grid on
xlabel("x")
ylabel('training set RMSE [$\%$]')


%% Validation on some points randomly extracted in time domain
clear input_val output_real output_net_val_time output_real_time RMSE_val
x_val_index = randperm(1024,length(x_train_index));
x_val_index = sort(x_val_index);
x_val_index = x_val_index(x_val_index ~= x_train_index);
x_val_index = x_val_index(x_val_index>=5 & x_val_index<1024);

for jj = 1:length(x_val_index)
    x_pos = x_val_index(jj)*ones(length(tsave)-1,1);
    k = x_val_index(jj);
    input_val = [usave(1:end-1,k),ux(1:end-1,k),uxx(1:end-1,k),uxxxx(1:end-1,k),x_pos]; 
    output_real = [usave(2:end,k)];

    output_net_val_time(:,jj) = net(input_val.');
    output_real_time(:,jj) = output_real;
end  


figure; hold on; grid on
for kk = [20 50 80]
    plot(tsave(2:end),output_real_time(:,kk),'color',grey,'LineWidth',1.5,'DisplayName',"KS x="+num2str(x_val_index(kk)))
    plot(tsave(2:end),output_net_val_time(:,kk),'--','LineWidth',1.5,'DisplayName',"NN x="+num2str(x_val_index(kk)))
end
xlabel('time [s]')
ylabel("u")
legend show

for jj = 1:length(x_val_index)
    RMSE_val(jj) = rms(output_real_time(:,jj)-output_net_val_time(:,jj))/rms(output_real_time(:,jj))*100;
end
figure; 
plot(x_val_index,RMSE_val,'*','LineWidth',1.5)
grid on
xlabel("x")
ylabel('validation set RMSE [$\%$]')


%% Reconstruction entire grid
clear input_val output_real output_net_val_time output_real_time 
clear RMSE_total RMSE_val RMSE_train x_val_index

x_train_index = 5:8:1024;
x_space=5:length(xsave);
    % entire dataset
for jj = 1:length(x_space)
    x_pos = x_space(jj)*ones(length(tsave)-1,1);
    k = x_space(jj);
    input_val = [usave(1:end-1,k),ux(1:end-1,k),uxx(1:end-1,k),uxxxx(1:end-1,k),x_pos]; 
    output_real = [usave(2:end,k)];

    output_net_val_time(:,jj) = net(input_val.');
    output_real_time(:,jj) = output_real;
end  
output_net_total = [usave(1,5:end); output_net_val_time];

    % surf output ode vs output NN
t0 = tsave(1);
figure;
sp(1) = subplot(1,2,1);
 surf(tsave-t0,x_space,usave(:,x_space)'),shading interp, colormap("parula"), view(2);
 xlim([0 10]); ylim([0 1100])
 xlabel('time [s]'); 
 ylabel('x coordinate'); 
 c=colorbar; zlabel('output u KS equation')
 c.Label.String = "output u equation"; c.Label.FontSize = 16.5
 c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
sp(2) = subplot(1,2,2);
 surf(tsave-t0,x_space,output_net_total'),shading interp, colormap("parula"), view(2);
 xlim([0 10]); ylim([0 1100])
 xlabel('time [s]'); 
 ylabel('x coordinate'); 
 c= colorbar; zlabel('output u NN')
 c.Label.String = "output u NN"; c.Label.FontSize = 16.5
 c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';

    % rmse distincted between training and validation set
pp = 1; vv=1;
for jj = 1:length(x_space)
    RMSE_tot(jj) = rms(output_real_time(:,jj)-output_net_val_time(:,jj))/rms(output_real_time(:,jj))*100;
    if find(x_train_index==x_space(jj),1)
            RMSE_train(pp) = RMSE_tot(jj); pp=pp+1;
    else
            RMSE_val(vv) = RMSE_tot(jj);
            x_val_index(vv) = x_space(jj); vv=vv+1;
    end
end

    % plot rmse
figure; 
sp(1) = subplot(1,2,1);
plot(x_train_index,RMSE_train,'*','color',grey,'LineWidth',1.5,'MarkerSize',5)
xlabel('x coordinate');
ylabel('RMSE training set [$\%$]')
ylim([0 3])
sp(2) = subplot(1,2,2);
plot(x_val_index,RMSE_val,'*','LineWidth',1,'MarkerSize',5)
xlabel('x coordinate'); 
ylabel('RMSE validation set [$\%$]')
ylim([0 3])
linkaxes(sp,'x')
xlim([0 1100])


