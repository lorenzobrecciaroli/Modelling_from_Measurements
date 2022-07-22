clc
clear all

%% Dataset creation
x1_raw = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 ...
      137 18 22 52 83 18 10 9 65];
x2_raw = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 ...
      18 37 50 35 12 12 25];
X_raw = [x1_raw; x2_raw];

time_raw = 1845:2:1903;

grey = [0.4 0.4 0.4];

ALL_PLOT = 0;

%% Derivative computation
X = X_raw;
[m,n] = size(X);
ind = n-1; 
dt = 1;
Xdot= zeros(m,ind);
for kk = 1:m %explore rows = animal
    for jj = 2:ind %explore timeseries
        Xdot(kk,jj-1) = (X(kk,jj+1)-X(kk,jj-1))/2/dt;
    end
end

%% Least squares fitting
X_red = X(:,2:ind+1);

X1 = X_red(1,1:end)';
X2 = X_red(2,1:end)';
X1X2 = X1.*X2;
O = zeros(ind,1);

A = [X1 -X1X2 O O; O O X1X2 -X2];
Xdot_tmp = [Xdot(1,:)'; Xdot(2,:)'];

% phi_hat = pinv(A)*Xdot_tmp; %estimated parameter vectors
phi = A\Xdot_tmp; % closed-loop formula linear regression in the unknown parameters
% phi_hat3 = lsqr(A,Xdot_tmp);


%% Model simulation
Xdot_hat = A*phi;
Xdot1_hat = Xdot_hat(1:ind)';
Xdot2_hat = Xdot_hat(ind+1:end)';

pp = length(Xdot2_hat);
figure; hold on;
plot(time_raw(1:pp),Xdot(1,:),'-*','DisplayName','Hare')
plot(time_raw(1:pp),Xdot1_hat(1,:),'-*','DisplayName','L-V Hare')
plot(time_raw(1:pp),Xdot(2,:),'-o','DisplayName','Lynx')
plot(time_raw(1:pp),Xdot2_hat(1,:),'-o','DisplayName','L-V Lynx')
legend show
xlabel('time [years]')
ylabel('Xdot')


if ALL
    clear x1_hat x2_hat
    x1_hat(1) = x1_raw(1);
    x2_hat(1) = x2_raw(1);
    Ts = dt;
    % 
    for kk = 1:length(x1_raw)-1
        x1_hat(kk+1) = x1_hat(kk) + Ts*(phi(1)*x1_hat(kk) - phi(2)*x1_hat(kk)*x2_hat(kk));
        x2_hat(kk+1) = x2_hat(kk) + Ts*(phi(3)*x1_hat(kk)*x2_hat(kk) -phi(4)*x2_hat(kk));
    end
else
    % ---- 1 step only prediction ----
    clear x1_hat x2_hat
    x1_hat(1) = x1_raw(1);
    x2_hat(1) = x2_raw(1);
    Ts = dt;
    for kk = 1:length(x1_raw)-1
        x1_hat(kk+1) = x1_raw(kk) + Ts*(phi(1)*x1_raw(kk) - phi(2)*x1_raw(kk)*x2_raw(kk));
        x2_hat(kk+1) = x2_raw(kk) + Ts*(phi(3)*x1_raw(kk)*x2_raw(kk) -phi(4)*x2_raw(kk));
    end
end

figure; 
subplot(1,2,1)
hold on; grid on;   
plot(time_raw,x1_raw,'-*','color',grey,'DisplayName','Hare','MarkerSize',4)   
plot(time_raw,x1_hat,'-*','DisplayName','LV Hare','MarkerSize',4)   
legend('show');
xlabel('time [year]'); ylabel('head numbers [thousands]')

subplot(1,2,2)
hold on; grid on;  
plot(time_raw,x2_raw,'-*','color',grey,'DisplayName','Lynx','MarkerSize',4)   
plot(time_raw,x2_hat,'-*','DisplayName','LV Lynx','MarkerSize',4)
legend('show');
xlabel('time [year]');
