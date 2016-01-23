punctuation = ['!','"','#','$','%','&','\'','(',')','*','+',',',
'-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`',
'{','|','}','~']

function clean(string)
  string = Base.lowercase(string)
  sarray = convert(Array{typeof(string)}, Base.split(string))
  x = 1 

  for x in 1:length(sarray)
    word = join(split(sarray[x], char(0x2223)))
    sarray[x] = word
  end
  x=1

  while x <= length(sarray)
    sarray[x] = Base.strip(sarray[x], punctuation)
    x += 1
  end

  return sarray
end

function parse_xml(doc_path)
  xdoc = parse_file(doc_path)
  xroot = root(xdoc)
  ces = get_elements_by_tagname(xroot, "EEBO")
  children = collect(child_elements(ces[1]))
  if(name(children[2]) == "GROUP")
    ces = get_elements_by_tagname(ces[1], "GROUP")
  else
    ces = get_elements_by_tagname(ces[1], "TEXT")
  end
  body = content(ces[1])
  text = string(body)

  free(xdoc)
  
  return text
end

function get_metadata(doc_path)
  metadata = Array(String,4)
  xdoc = parse_file(doc_path)
  xroot = root(xdoc)
  ces = get_elements_by_tagname(xroot, "HEADER")
  filedesc = get_elements_by_tagname(ces[1], "FILEDESC")
  source = get_elements_by_tagname(filedesc[1], "SOURCEDESC")
  bib = get_elements_by_tagname(source[1], "BIBLFULL")
  titlestmt = get_elements_by_tagname(bib[1], "TITLESTMT")
  author = get_elements_by_tagname(titlestmt[1], "AUTHOR")
  
  if length(author) == 0
    metadata[1] = "NA"
  else
    author = content(author[1])
    author = string(author)
    metadata[1] = author
  end

  title = get_elements_by_tagname(titlestmt[1], "TITLE")
  if length(title) == 0
    metadata[2] = "NA"
  else
    title = content(title[1])
    title = string(title)
    metadata[2]  = title
  end

  date = get_elements_by_tagname(bib[1], "PUBLICATIONSTMT")
  date = get_elements_by_tagname(date[1], "DATE")
  if length(date) == 0
    metadata[3] = "NA"
  else
    date = content(date[1])
    date = string(date)
    metadata[3] = date
  end

  profile = get_elements_by_tagname(ces[1], "PROFILEDESC")
  langusage = get_elements_by_tagname(profile[1], "LANGUSAGE")
  lang = get_elements_by_tagname(langusage[1], "LANGUAGE")
  if length(lang) == 0
    metadata[4] = "NA"
  else
    lang = content(lang[1])
    lang = string(lang)
    metadata[4] = lang
  end
  
  free(xdoc)

  return metadata
end
  

function load_featurevector(path)
  if isfile(path)
    fv = FeatureVector()
    parsed_file = parse_xml(path)
    clean_text = clean(parsed_file)
    for word in clean_text
      if word in keys(fv)
        fv[word] += 1
      else
        setindex!(fv,1,word)
      end
    end
    fv[""] = 0
    fv.mdata = get_metadata(path)
    return fv
  else
    Base.warn("$path is not to a valid file. Please check the path and try again.")
  end
end

function load_cluster(path)
  if isdir(path)
    curdir = pwd()
    cd(path)
    files = readdir(pwd())
    cl = Cluster()
    for file in files
      if endswith(file,"xml")
        fv = load_featurevector(file)
        cl[file] = fv
      else
        Base.warn("$file is not a valid file. Please enter a path to a directory of .xml files.")
      end
    end
    cd(curdir)
    return cl
  else
    Base.warn("$path is not to a valid directory. Please check the path and try again.")
  end
end

function load_dataset(path)
  if isdir(path)
    curdir = pwd()
    cd(path)
    dirs = readdir(pwd())
    ds = DataSet()
    for dir in dirs
      cl = Cluster()
      if isdir(dir)
        cl = load_cluster(dir)
        ds[dir] = cl
      else
        Base.warn("$dir is not a valid Directory. This will not be added.")
      end
    end
    cd(curdir)
    return ds
  else
    Base.warn("$path is not to a valid directory. Please check the path and try again.")
  end
end
