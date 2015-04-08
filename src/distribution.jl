type Distribution{FS<:FeatureSpace}
  space::FS
  features::Number
  total::Number
  smooth::Function
  smooth_data::Array
  mdata::Any
  Distribution() = new(FS(),0,0,_no_smoothing,[])
  Distribution(fv::FeatureVector) = new(fv,length(fv),get_total(fv),_no_smoothing,[])
  Distribution(c::Cluster) = new(c,length(c.vector_sum),get_total(c.vector_sum) ,_no_smoothing,[])
  Distribution(ds::DataSet) = new(ds,length(ds.vector_sum),get_total(ds.vector_sum) ,_no_smoothing,[])
  function get_total(fv::FeatureVector)
    total = 0
    for value in values(fv)
      total += value
    end
    return total
  end
end
Distribution(fv::FeatureVector) = Distribution{FeatureVector}(fv)
Distribution(c::Cluster) = Distribution{Cluster}(c)
Distribution(ds::DataSet) = Distribution{DataSet}(ds)

function getindex(d::Distribution, key)
  return d.space[key]
end

function probability(d::Distribution, feature)
  return d.smooth(d, feature, d.smooth_data)
end

function keys(d::Distribution)
  return keys(d.space)
end

function features(d::Distribution{FeatureVector})
  return keys(d.space)
end

function features(d::Distribution)
  return keys(d.space.vector_sum)
end

function isempty(d::Distribution)
  return isempty(d.space)
end

function entropy(d::Distribution)
  ent = 0
  for feature in features(d)
    ent -= probability(d,feature)*log2(probability(d,feature))
  end
  return ent
end

function info_gain(d1::Distribution, d2::Distribution)
  return entropy(d1)-entropy(d2)
end

function perplexity(d::Distribution)
  return 2^entropy(d)
end

function display(dist::Distribution)
  display(dist.space)
end

#helper function that sets the smoothing type
function set_smooth!(d::Distribution{FeatureVector}, f::Function, sd::Array)
  d.smooth = f
  d.smooth_data = sd
end

#no smoothing default
function remove_smoothing!(d::Distribution)
  set_smooth!(d,_no_smoothing,[])
end

function _no_smoothing(d::Distribution{FeatureVector}, key, data::Array)
  return d.space[key] / d.total
end

function _no_smoothing(d::Distribution, feature, data::Array)
  return d.space.vector_sum[feature] / d.total
end

#add-delta smoothing, defaults to add-one smoothing
function delta_smoothing!(d::Distribution, δ::Number=1)
  if δ <= 0
    Base.warn("δ must be greater than 0") 
  end
  set_smooth!(d,_δ_smoothing,[δ,d.features,d.total])
end

function _δ_smoothing(d::Distribution{FeatureVector}, key, data::Array)
  if !haskey(d.space, key)
    return (data[1])/(data[1]*(data[2]+1)+data[3])
  end
  return (d.space[key]+data[1])/(data[1]*(data[2]+1)+data[3])
end

function _δ_smoothing(d::Distribution, key, data::Array)
  if !haskey(d.space.vector_sum, key)
    return (data[1])/(data[1]*(data[2]+1)+data[3])
  end
  return (d.space[key]+data[1])/(data[1]*(data[2]+1)+data[3])
end

#good turing count adjust, no smoothing for unseen frequencies
function goodturing_smoothing!(d::Distribution{FeatureVector})
  freqs = FeatureVector() 
  for value in values(d.space)
    freqs[value] += 1
  end
  set_smooth!(d,_gt_smoothing, [d.total, freqs])
end

function goodturing_smoothing!(d::Distribution)
  freqs = FeatureVector() 
  for value in values(d.space.vector_sum)
    freqs[value] += 1
  end
  set_smooth!(d,_gt_smoothing, [d.total, freqs])
end

function _gt_smoothing(d::Distribution{FeatureVector}, key, data::Array)
  if !haskey(d.space, key)
    return data[2][1] / data[1] #num of keys that occur once / total number of keys
  end
  c = d.space[key]
  c_adjust = (c+1) * (data[2][c+1]/data[2][c])
  return c_adjust / data[1]
end

function _gt_smoothing(d::Distribution, key, data::Array)
  if !haskey(d.space.vector_sum, key)
    return data[2][1] / data[1] #num of keys that occur once / total number of keys
  end
  c = d.space.vector_sum[key]
  c_adjust = (c+1) * (data[2][c+1]/data[2][c])
  return c_adjust / data[1]
end

#good-turing smoothing without tears
#by gale, at&t labs
function simplegoodturing_smoothing!(d::Distribution)
  freqs = FeatureVector() 
  for frequency in values(d.space.vector_sum)
    freqs[frequency] += 1
  end
  iter = freq_list(freqs, (a,b) -> a[1]<b[1])
  stop = length(freqs)
  x = zeros(stop)
  y = zeros(stop)
  Z = FeatureVector()
  Z[1] = convert(FloatingPoint, freqs[1])
  for (i, pair) in enumerate(iter)
    if !done(iter, i+1)
      t = iter[i+1][1]
    end
    if (i != start(iter)) 
      Z[pair[1]] = freqs[pair[1]] / (0.5*(t-q))
    end
    q = pair[1]
    x[i] = log(pair[1])
    y[i] = log(Z[pair[1]])
  end
  a, b = linreg(x,y)
  if b < -1 
    set_smooth!(d,_sgt_smoothing, [d.total, Z, a, b])
  else 
    set_smooth!(d,_gt_smoothing, [d.total, freqs])
  end
end

function _sgt_smoothing(d::Distribution, key, data::Array)
  if !haskey(d.space.vector_sum, key)
    return data[2][1] / data[1]
  end
  #check to see which values of r using _gt_ value and when to switch to smooth
end