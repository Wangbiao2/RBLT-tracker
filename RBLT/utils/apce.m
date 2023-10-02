function APCE = apce(response)

    Fmax = max(response(:));
    Fmin = min(response(:));
    Sum = (response-Fmin).^2;
    Fmean =  mean(Sum,'All');  
    APCE = (Fmax - Fmin)^2 / Fmean;
end

