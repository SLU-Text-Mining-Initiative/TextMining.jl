using FactCheck


export Distribution

println("**********Testing Distribution**********")

facts("Creating a Distribution") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)
	d2 = Distribution()

	for key in keys(fv1)
		@fact d1.fv[key] => fv1[key]
		@fact d1[key] => not(fv1[key])
	end
	@fact typeof(d1.fv) => typeof(fv1)
	@fact typeof(d2.fv) => typeof(FeatureVector{Any,Number}())
end

facts("If sent in with a FeatureVector, d.total is set") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)
	d2 = Distribution()

	@fact d1.total => 7
	@fact d2.total => 0
end

facts("Get probability of seeing key in Distribution") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)

	value = d1["word"]

	@fact value => 4/7
end

facts("Set a value given key") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)

	d1["word"] = 7

	@fact d1.fv["word"] => 7
end

facts("Modifying FeatureVector of Distribution does not modify original FeatureVector") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)

	d1["word"] = 7

	@fact d1.fv["word"] => 7
	@fact fv1["word"] => 4
end

facts("Set a value will change Distribution.total") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)

	@fact d1.total => 7
	d1["word"] = 7
	@fact d1.total => 10
end

facts("Get keys of Distribution") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)

	@fact keys(d1) => Base.keys(d1.fv.map)
end

facts("Get values of Distribution") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)

	@fact values(d1) => Base.values(d1.fv.map)
end

facts("Check isempty on a Distribution") do
	d1 = Distribution()

	@fact isempty(d1) => Base.isempty(d1.fv.map)
end

facts("entropy(Distribution) returns 0 if empty Distribution") do
	d1 = Distribution()

	@fact entropy(d1) => 0
end

facts("entropy(Distribution) returns entropy of Distribution") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)

	@fact entropy(d1) => 0.9852281360342515
end

facts("info_gain(Distribution,Distribution) returns 0 if empty Distributions") do
	d1 = Distribution()
	d2 = Distribution()

	@fact info_gain(d1,d2) => 0
end

facts("info_gain(Distribution,Distribution) returns info_gain of Distribution") do
	dict1 = ["word" => 4, "another" => 3]
	fv1 = FeatureVector(dict1)
	d1 = Distribution(fv1)

	dict2 = ["∂" => .1, "happy" => .3]
	fv2 = FeatureVector(dict2)
	d2 = Distribution(fv2)

	@fact info_gain(d1,d2) => 0.17395001157511858
end
