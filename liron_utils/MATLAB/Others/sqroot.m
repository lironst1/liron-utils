% Find square root using recursive function


function sqroot(A, g, t)

g = find_sqroot(A, g, t);

figure()
plot(g, "Marker", "o")


    function g = find_sqroot(A, g, t)

        if abs(g(end)^2 - A) < t
            return
        end

        g(end+1) = 0.5*(g(end) + A/g(end));

        g = find_sqroot(A, g, t);

    end
end
