clear 
close all
clc

load("reaction_diffusion_big.mat");

figure
imagesc(u(:,:,1));

figure
imagesc(v(:,:,1));


%% Datamatrix X creation
clear x_tmp x_col
X = [];
x_data = 250:350;
y_data = 250:350;

for k = 1:length(t)
    k
    x_col = [];
    for xx = x_data
        for yy = y_data
            x_tmp = [u(xx,yy,k);v(xx,yy,k)];
            x_col = [x_col;x_tmp];            
        end
    end

    X = [X,x_col];
end

[coeff,score,~,~,explained,mu] = pca(X');

idx = find(cumsum(explained)>95,1);

Z_red = score(:,1:idx);


% --- Construct the input for the NN ---
t_train_index = 140;
input = Z_red(1:t_train_index,:);
output = Z_red(2:t_train_index+1,:);



%% Training NN
TRAIN = 0;
if TRAIN
    net = feedforwardnet([10 10]);
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.transferFcn = 'radbas';
    net.trainParam.epochs = 1500;
    net.trainParam.max_fail = 1000;
    [net,tr] = train(net,input.',output.');
end

%% Compute output in validation and reproject in original space
load("RD_NeuralNetwork.mat"); % NN trained
x_data = 250:350;
y_data = 250:350;
t_validation = 170;
 
z_val_in = (X(:,t_validation)'- mu)*coeff(:,1:idx);

z_val_out = net(z_val_in');

x_val_out = z_val_out'*coeff(:,1:idx)' + mu;
x_real = X(:,t_validation)';

u_val = zeros(size(u(:,:,1)));
v_val = zeros(size(v(:,:,1)));
k = 1;
for xx = x_data
    for yy = y_data
        u_val(xx,yy) = x_val_out(k);
        k = k+1;
        v_val(xx,yy) = x_val_out(k);
        k = k+1;           
    end
end

figure;
subplot(2,2,1)
surf(x_data,y_data,u(x_data,y_data,t_validation + 1)),shading interp, colormap("parula"), view(2);
 c=colorbar; zlabel('output u equation')
 c.Label.String = "output u equation"; c.Label.FontSize = 9
 c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
 xlabel('x coordinate');  ylabel('y coordinate'); 

subplot(2,2,2)
surf(x_data,y_data,u_val(x_data,y_data)),shading interp, colormap("parula"), view(2);
 c=colorbar; zlabel('output u NN')
 c.Label.String = "output u NN"; c.Label.FontSize = 9
 c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
 xlabel('x coordinate');  ylabel('y coordinate'); 

subplot(2,2,3)
surf(x_data,y_data,v(x_data,y_data,t_validation + 1)),shading interp, colormap("parula"), view(2);
 c=colorbar; zlabel('output v equation')
 c.Label.String = "output v equation"; c.Label.FontSize = 9
 c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
 xlabel('x coordinate');  ylabel('y coordinate'); 

subplot(2,2,4)
surf(x_data,y_data,v_val(x_data,y_data)),shading interp, colormap("parula"), view(2);
 c=colorbar; zlabel('output v NN')
 c.Label.String = "output v NN"; c.Label.FontSize = 9
 c.Label.Interpreter = 'latex'; c.TickLabelInterpreter = 'latex';
 xlabel('x coordinate');  ylabel('y coordinate'); 

%% Average error computation on the entire grid
dt = 0.05;
u_error = [];
v_error = [];
t_select = 140:length(t)-1;
for t_validation = t_select

    z_val_in = (X(:,t_validation)'- mu)*coeff(:,1:idx);
    
    z_val_out = net(z_val_in');
    
    x_val_out = z_val_out'*coeff(:,1:idx)' + mu;
    x_real = X(:,t_validation+1)';

    u_error = [u_error,mean(abs((x_real(1:2:end-1)-x_val_out(1:2:end-1))./abs(x_real(1:2:end-1))),'all')];
    v_error = [v_error,mean(abs((x_real(2:2:end)-x_val_out(2:2:end))./abs(x_real(2:2:end))),'all')];
end    


figure
subplot(1,2,1)
plot(t_select,u_error.*100,'*'); grid on; box on
xlabel('time [s]')
ylabel('average grid error on u [%]'); ylim([0,10])
subplot(1,2,2)
plot(t_select,v_error.*100,'*'); grid on; box on
xlabel('time [s]')
ylabel('average grid error on v [%]'); ylim([0,10])



