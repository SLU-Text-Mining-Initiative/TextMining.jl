# TextMining.jl 



[![Build Status](https://travis-ci.org/SLU-TMI/TextMining.jl.svg?branch=master)](https://travis-ci.org/SLU-TMI/TextMining.jl)

This package is a set of tools being used by Saint Louis University to facilitate interdisciplinary research  of how time passage affects language using data mining, machine learning, and natural language processing techniques.
---
For further information, contact project leader Lauren Kersey at kersey@slu.edu.
---
<div align="center">
<a href="http://julialang.org/" target="_blank">
<img src="http://julialang.org/images/logo_hires.png" alt="Julia Logo" width="105" height="71"></img>
</a>
</div>

These tools are being developed in **Julia** with the goal of making them fast, generic, and easily usable in Julia's REPL. Installing the *TextMining* package is done like so:

```julia
Pkg.clone("https://github.com/SLU-TMI/TextMining.jl.git")
```
---
### Table of Contents
1. **[Feature Space Model](https://github.com/SLU-TMI/TextMining.jl#feature-space-model)**
  1. **[Feature Vector](https://github.com/SLU-TMI/TextMining.jl#feature-vector)**
  2. **[Cluster](https://github.com/SLU-TMI/TextMining.jl#cluster)**
  3. **[DataSet](https://github.com/SLU-TMI/TextMining.jl#dataset)**
2. **[Clustering](https://github.com/SLU-TMI/TextMining.jl#clustering)**
  1. **[k Means](https://github.com/SLU-TMI/TextMining.jl#k-means)**
  2. **[Hierarchical](https://github.com/SLU-TMI/TextMining.jl#hierarchical)**
3. **[Classification](https://github.com/SLU-TMI/TextMining.jl#classification)**
  1. **[Proximity Based Classification](https://github.com/SLU-TMI/TextMining.jl#proximity-based-classification)**
    1. **[k Nearest Neighbors](https://github.com/SLU-TMI/TextMining.jl#k-nearest-neighbors)**
  2. **[Probability Based Classification](https://github.com/SLU-TMI/TextMining.jl#probability-based-classification)**
    1. **[Distribution](https://github.com/SLU-TMI/TextMining.jl#distribution)**
    2. **[Naive Bayes](https://github.com/SLU-TMI/TextMining.jl#naive-bayes)**
4. **[Text Processing](https://github.com/SLU-TMI/TextMining.jl#text-processing)**

---
Feature Space Model
------

These tools will utilize the [bag-of-words model](http://en.wikipedia.org/wiki/Bag-of-words_model) and the [hashing trick](http://en.wikipedia.org/wiki/Feature_hashing) to vectorize texts into [feature vectors](http://en.wikipedia.org/wiki/Feature_vector). Feature vectors exist in an infinite dimensional vector space which is refered to as the **feature space**. In order to optimize calculations, dimensions where the feature vector has value 0 are removed from the feature vector's hashtable. We are defining **FeatureSpace** to be an abstract type which has 3 subtypes: FeatureVector, Cluster, and DataSet.


#### Feature Vector

The **FeatureVector** type is a container for a **Dictionary (hashtable)** that restricts **key => value** mappings to **Any => Number** mappings, where *Any* and *Number* are Julia types, or their subtypes. 

##### Using FeatureVector:

Constructing an empty *FeatureVector*:
```julia
julia> fv = FeatureVector()
```

A *FeatureVector* can also be constructed using an existing *Dictionary*: 
```julia
julia> dict = ["word1"=>2, "word2"=>4]
julia> fv = FeatureVector(dict)
```
or
```julia
julia> fv = FeatureVector(["word1"=>2, "word2"=>4])
```

Modifying elements of a *FeatureVector*:
```julia
julia> fv["word1"] = 4
```

Accessing elements of a *FeatureVector*:
```julia
julia> fv["word1"]
4
```

Addition and subtraction between two *FeatureVectors*:
```julia
julia> fv1 = FeatureVector(["word1" => 2, "word2" => 4])
julia> fv2 = FeatureVector(["word1" => 4, "word2" => 2])
julia> fv1+fv2
FeatureVector{ASCIIString,Int64}(["word1"=>6,"word2"=>6])

julia> fv1-fv2
FeatureVector{ASCIIString,Int64}(["word1"=>-2,"word2"=>2])
```

Multiplication and division by a scalar:
```julia
julia> fv = FeatureVector(["word1" => 1, "word2" => 3])
julia> fv*3
FeatureVector{ASCIIString,Int64}(["word1"=>3,"word2"=>9])

julia> fv/3
FeatureVector{Any,Float64}(["word1"=>0.3333333333333333,"word2"=>1.0])
```

If a *FeatureVector* contains Integer value types it can be rationalized by a divisor:
```julia
julia> fv//3
FeatureVector{ASCIIString,Rational{T<:Integer}}(["word1"=>1//3,"word2"=>1//1])
```
but returns an error otherwise:
```julia
julia> fv = FeatureVector(["word1" => 1.0, "word2" => 3.0])
julia> fv//3
ERROR: `//` has no method matching //(::Float64, ::Int64)
```
---
##### FeatureVector Functions:
```julia
keys(fv)
```
* Returns an **Iterator** to the keys in **fv**.
```julia
values(fv)
```
* Returns an **Iterator** to the values in **fv**.
```julia
haskey(fv, key)
```
* Checks **fv** for **key** and returns **true** if found or **false** if not present.
```julia
isempty(fv)
```
* Returns **false** if **fv** contains any elements, **true** otherwise.
```julia
length(fv)
```
* Returns the number of elements in **fv**.
```julia
freq_list(fv, expression = (a,b) -> a[2]>b[2])
```
* Returns a [frequency list](http://en.wikipedia.org/wiki/Word_lists_by_frequency) represented by an **Array** of (key,value) tuples sorted using the provided boolean expression. If an **expression** is not passed in, the Array will be sorted by largest value.
```julia
add!(fv1, fv2)
```
* In place addition. Modifies **fv1** by adding **fv2** to it. 
```julia
subtract!(fv1, fv2)
```
* In place subtraction. Modifies **fv1** by subtracting **fv2** from it. 
```julia
multiply!(fv, value)
```
* In place multiplication. Scales **fv** by value. 
```julia
divide!(fv, value)
```
* In place division. Divides **fv** by value.
```julia
rationalize!(fv, value)
```
* In place rationalization. Rationalizes **fv** by value.
```julia
dist_cos(fv1, fv2)
```
* Returns 1-[cosine similarity](http://en.wikipedia.org/wiki/Cosine_similarity) between two feature vectors. If the angle between **fv1** and **fv2** is 0, the function will return 0. If **fv1** and **fv2** are orthogonal, meaning they share no features, the function will return 1. Otherwise the function returns values between 0 and 1. **Note:** The zero vector is both parallel and orthogonal to every vector, as such **cos\_dist(fv, zero_vector)** will return **NaN** (not a number). 
```julia
dist_zero(fv1, fv2)
```
* Derived from the L0 Norm, this function returns the number of non-zero elements that differ between **fv1** and **fv2**. 
```julia
dist_taxicab(fv1, fv2)
```
* Derived from the [L1 Norm](http://en.wikipedia.org/wiki/Taxicab_geometry) and also know as the Manhattan distance, this function returns the sum of the absolute difference between **fv1** and **fv2**. 
```julia
dist_euclidean(fv1, fv2)
```
* Returns the [standard distance](http://en.wikipedia.org/wiki/Euclidean_distance) between **fv1** and **fv2**. 
```julia
dist_infinite(fv1, fv2)
```
* Derived from the [L∞ Norm](http://en.wikipedia.org/wiki/Chebyshev_distance) and often referd to as the Chebyshev distance, this function returns the maximum absolute difference between any feature in **fv1** or **fv2**.

---

#### Cluster

The **Cluster** type is also a **Dictionary** container. However, it restricts mappings to **Any => FeatureVector** types and subtypes. This allows users to meaningfully label groups of *FeatureVectors* for Classification. The *Cluster* type also computes the centroid of the set.

An empty *Cluster* can be constructed as so:
```julia
cl = Cluster()
```
---
#### DataSet

The **DataSet** type is also a wrapper around a **Dictionary**. However, it restricts mappings to **Any => Cluster** types and subtypes.

An empty *DataSet* can be constructed as so:
```julia
ds = DataSet()
```
---
Clustering
------

#### The simplest way to run kmeans clustering is to enter the data as a "cluster" type and predefine the number of categories.  
```julia
julia> results = kmeans(cluster.vectors, k)
```


#### Hierarchical

---
Classification
------

### Proximity Based Classification
---

#### k Nearest Neighbors

---
### Probability Based Classification
---

#### Distribution

The **Distribution** type is a container which ensures the [axioms of probability](http://en.wikipedia.org/wiki/Probability_axioms).

An empty *Distribution* can be constructed as so:
```julia
ds = Distribution()
```

#### Naive Bayes
---
Text Processing
------

#### Processing XML Files
---
