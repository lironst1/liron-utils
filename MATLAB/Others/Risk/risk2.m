function [aa ab da db]=risk2(a,b)

dbstop if error


% a=23               % attacker
% b=17               % defender
da=0;
db=0;
ca=[];
cb=[];
aa=a;
bb=b;

while 1

    if min(length(ca),length(cb)) <= 0
        % roll dice

        if aa >= 4
            ca=cube(1,3);
        elseif aa == 3
            ca=cube(1,2) ;
        elseif   aa==2
            ca=cube(1,1);
        else
            disp('A lose')
            disp(['dead:      A:     ',num2str(da) '  dead: B:   :     ',num2str(db)])
            disp(['Alive:   A:     ',num2str(aa) '  Alive B:      ',num2str(ab)])
            disp('  ')
            pause
            break
        end



        if bb >= 2
            cb=cube(1,2);
        elseif bb == 1
            cb=cube(1,1) ;
        else
            disp('B lose')
            disp(['dead:      A:     ',num2str(da) '  dead: B:   :     ',num2str(db)])
            disp(['Alive:   A:     ',num2str(aa) '  Alive B:      ',num2str(ab)])
            disp('  ')
            pause
            break
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
    if  min(length(ca),length(cb)) <= 1
        %         [da db]

        disp(['dead:      A:     ',num2str(da) '  dead: B:   :     ',num2str(db)])
        disp(['Alive:   A:     ',num2str(aa) '  Alive B:      ',num2str(ab)])
        disp('  ')
        pause
    end



    if aa < 2 || ab <= 0
        disp ('Battle Ended')
        PlayGong
        return
    end


    [m,k]=max(ca);
    ca(k)=[];
    [m,k]=max(cb);
    cb(k)=[];


end


