function APCE = apce(response)
    Fmax= max(max(response));
%     disp(Fmax)
    Fmin = min(min(response));
    
    Sum = (response-Fmin).^2;
%     for i = 1: size(response,1)
%         for j = 1 :size(response,2)
%             Sum = Sum +( response(i,j) - Fmin)^2;
%               
%         end
%     end
    
    Fmean =  mean(Sum,'All');  
    
   
    APCE = (Fmax - Fmin)^2 / Fmean;
end

