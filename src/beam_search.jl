
function beam_search(max_features)

    for i in 1:max_features
        expanded_set = Set()
        expanded_arr = [] #TODO: optimize to one alloc
        for cand in top_k
            expanded = expand_all(cand, inst)
            for subset in expanded
                if !(subset in expanded_set)
                    push!(expanded_set, subset)
                    append!(expanded_arr, [subset])
                end
            end
        end
        num_expanded = size(expanded_arr)[1]
        expanded_ep = nothing
        expanded_mars = nothing
        if encoder === nothing
            expanded_ep = ep_func(expanded_arr, ep_params, true, logger)
            expanded_mars = marginals(expanded_arr, ep_params[1])
        else
            expanded_ep = ep_func(encoder(expanded_arr,cats), ep_params, true, logger)
            expanded_mars = marginals(encoder(expanded_arr,cats), ep_params[1])
        end
        expanded_cands = Array{Tuple{Candidate,Float64}}(undef,num_expanded)
        for j in 1:num_expanded
            expanded_cands[j] = (Candidate(expanded_arr[j], expanded_ep[j]),expanded_mars[j])
        end
        if (predicted_label == 1)
            sort!(expanded_cands, by = x -> (x[1].prob, x[2]),rev=true)
        else
            sort!(expanded_cands, by = x -> (-x[1].prob,x[2]), rev=true)
        end
        top_k = []
        for j in 1:k
            if j > num_expanded
                break
            end
            append!(top_k, [expanded_cands[j][1].features])
        end
        best_level_ep = expanded_cands[1][1].prob
        best_level_mar = expanded_cands[1][2]
        if (predicted_label == 1 && best_level_ep > best_ep) || (predicted_label == 0 && best_level_ep < best_ep)
            best_ep = best_level_ep
            best_log_mar = best_level_mar
            best_cand = expanded_cands[1][1].features
        elseif (best_level_ep == best_ep && best_level_mar > best_log_mar)
            best_log_mar = best_level_mar
            best_cand = expanded_cands[1][1].features
        end
    end