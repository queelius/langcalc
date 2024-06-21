

# TODO

- skip token(s) (when doing suffix expansion). we can compare the similarity of the
  suffix with skipped tokens (when they do not exist in the training data):

  let input x = w_{<t}
  
  we can skip tokens in x, call it y:

  y = w_{<j} w_{j+k:t}
  
  so we skipped the subsequence w_{j:j+k-1}.
  how similar is x and y?

  we can do suffix extension by substituing semantically similar tokens too,
  e.g., word2vec.

  a more complicated approach: pattern matching. if we substitute a token with
  another token (ideally similar in meaning, like substitute "daschund" with "dog"),
  then we ought to substitute all occurences of "daschund" with "dog" in the
  suffix. moreover, when we predict future tokens, we should rename "dog"
  to "daschund" in the predicted tokens so that even though we operate over
  "dog" in the model, we can still output "daschund" in the final output.

  one can imagine *many* kinds of transformations that can be done to the input
  and output, patterns that can be matched, etc. this is a very rich space.

  this is what neural LLMs do, but in a much richer and learnable way.

  another inductive bias: imposing grammars on the output. if we, say, notice
  that the suffix is a python code snippet, we can impose python grammar on the
  output until we output a complete python code snippet. this is a very strong
  inductive bias, and can be very useful in many applications.

  
- discuss in context of solomonoff induction. the universal prior is an inductive bias: small programs given more weight than large programs.
	- designed in the context of a non-random DGP and uncertainty or randomness comes from the fact that of the countably infinite number
	  of programs compatible with the data so far, we don't know which program is the DGP.

- once we have the solomnooff idea in place, we focus on the simple n-gram (or *-gram, or infini-gram) AR model estimate of DGP. same idea
  as solomonoff induction: learn the DGP. but the space of programs we consider is much smaller, and is largely just the empirical distribution
  combined with some inductive biases to help it generalize OOD.

	- however, we expand the space of programs by concidering the AR model combined with the space of programs that rewrite the context.
          and in particular, the space of programs that *project* the input onto the training data.

- Set up a large infini-gram model for experimentation.

- Try different projections (representing different kinds of inductive biases)
	- input expansion (try variations of the input using classical IR), and of course for each such variation, use the longest prefix.
		- longest or maybe that too can be an a hyper-paramter: sample a length N ~  (0, longest), where it might
                  assign a non-uniform probability (this can be fine-tuned for the data / task)
		- for each input expansion, we now have a continuation. how do we combine these results? this is another
		  kind of aggregation model. many different possibilities here.
	- embeddings. use LLM context representations to find nearby data points in training data. we can combine this with input expansion.
		- as previously for input expansion, this can give multiple outputs. aggegrate in some way
	- see paper draft for more ideas


- let's discuss different kinds of inductive biases in the context of an n-gram model.

	- grammars. we can have it produce, for instance, python code. it's a well-defined grammar. the training data can include
	  a lot of python + english, but then we only produce python in the model. the python, when ran, of course can then
	  generate any other kind of data. but, that may or may not be the primary task. the primay task could just be
	  helping write code, as is already a popular usecase (copilot).

	- grammars covers a lot of space, and is a very restrictive kind of inductive bias. there must be enough code in the
	  training data so that the AR model can produce code for a number of different projected inputs. the variety and
	  quality and specificity of the training data will make a huge difference.

	- when we are selective with our training data, that too is an inductive bias. ideally, we give the model a task that
	  we want it to get good at, and it finds a way to select its data that it learns from.
