function [aa ab da db]=risk(a,b)

dbstop if error 


% a=23               % attacker 
% b=17               % defender 
da=0;
db=0;
ca=[];
cb=[];
while 1 
    
     if min(length(ca),length(cb)) <= 0
        % roll dice 
       
        if a >= 4
            ca=cube(1,3);
        elseif a == 3 
            ca=cube(1,2) ;
        elseif   a==2 
            ca=cube(1,1);
        else
            disp('A lose')
            continue
        end     
     
     
    if b >= 2
        cb=cube(1,2);
    elseif b == 1 
        cb=cube(1,1) ;
    else
        disp('B lose')
        continue
    end
   
    disp(ca)
    disp(cb)
  end 

    if max(ca) > max(cb)
        % a wins 1
        db=db+1;
    else
        da=da+1;
    end
    aa=a-da;
    ab=b-db;
    if min(length(ca),length(cb)) <= 1
%         [da db]
        
        disp(['dead:      A:     ',num2str(da) '  dead: B:   :     ',num2str(db)])        
        disp(['Alive:   A:     ',num2str(aa) '  Alive B:      ',num2str(ab)]) 
        disp('  ')
        pause
    end 
%     if min(aa,ab) <= 0 
%         return 
%     end    
%         
      
    [m,k]=max(ca);
    ca(k)=[];
    [m,k]=max(cb);
    cb(k)=[];
    
    
   end 
    

end 