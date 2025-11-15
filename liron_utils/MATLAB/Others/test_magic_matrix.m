% magic matrix

function tf = test_magic_matrix(M)
tf = false;

sz = size(M);
assert(sz(1) == sz(2))


unq = zeros(1, 2);

for i = 1 : 2
    s = sum(M, i);

    if length(unique(s)) ~= 1
        return
    end

    unq(i) = unique(s);
end

if length(unique(unq)) ~= 1
    return
end

if trace(M) == unq(1) && trace(flip(M)) == unq(1)
    tf = true;
end


end
