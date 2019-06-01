function [Aks,bks]=get_elt_arrays2D(xv,yv,invJks,detJks,ne,...
                                        elt2vert,a,f)             
bks=zeros(ne,3); Aks=zeros(ne,3,3);
dpsi_ds=[-1,1,0]; dpsi_dt=[-1,0,1]; % for r=1
for i=1:3
    for j=1:3
        grad=[dpsi_ds(i) dpsi_ds(j); dpsi_dt(i) dpsi_dt(j)];
        v1=squeeze([invJks(:,1,1:2)])*grad;
        v2=squeeze([invJks(:,2,1:2)])*grad;
        int=prod(v1,2)+prod(v2,2);
        Aks(:,i,j)=Aks(:,i,j)+a.*detJks.*int./2;
    end
    bks(:,i)=bks(:,i)+f.*detJks./6;   
end
