function U = refine_triangulation(Ps, us, Uhat)
for i=1:5
    r = compute_residuals(Ps, us, Uhat);
    J = compute_jacobian(Ps, Uhat);
    Uhat=Uhat-inv(transpose(J)*J)*transpose(J)*r;
end
U=Uhat;