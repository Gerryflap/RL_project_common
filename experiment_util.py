from abc import ABCMeta
from inspect import getcallargs, getargspec, getfullargspec, getargvalues, currentframe
import collections

import datetime
import tensorflow as tf

class Logger():
    def __init__(self):
        self.runs = dict()
        self.session_ref = "experiments_%s.log"%(datetime.datetime.now().isoformat())

    
    def start_experiment(self, configuration ):
        ref = "results_%s" % datetime.datetime.now().isoformat()
        self.runs[ref] =  "%s.log" % ref
            
        with open(self.session_ref, "a") as f:
            f.write( str({'log_file': self.runs[ref], 'configuration': configuration}))
            f.write("\n")
        return ref

    def log_result(self, ref, result ):
        with open(self.runs[ref], "a") as f:
            f.write( str(result) )
            f.write("\n")

class H5Logger():
    def __init__(self, session=None, configuration=None, filename="results.hdf5"):
        self.session = session or "session_%s" % ( datetime.datetime.now() )
        self.configuration = configuration
        self.h5_filename = filename
        self.h5 = h5py.File(filename, "a")
        #grp = f.create_group('%s' % session)
    
    def start_experiment(configuration = None):
        return H5Logger(session=self.session, configuration=configuration, filename=self.h5_filename)

    def result(self, res):
        if self.configuration is None:
            raise ValueError("A configuration was never initialized.")
        print("Writing experiment results to %s/%s" % (self.session, self.dataset_name))
        self.dataset_name = "results_%s" % (datetime.datetime.now ())
        self.dataset = self.h5.create_dataset("%s/%s" % (self.session, self.dataset_name), res.shape, dtype="f", compression="gzip")
        self.dataset.attrs['configuration'] = self.configuration
        self.dataset[:] = res


def configurable(init):
    '''
    This is a decorator function that catches the arguments applied to a function call.
    To define a function as the configuration function of an object: 
    @configurable
    __init__(self, *args, *kwargs)

    Default arguments are then made available in self.__defaultConfiguration__,
    and supplied arguments to the method are available in self.__configuration__.
    '''
    def dict_from_fn(fn):
        # the last values in init_args are arguments for which a default value is supplied
        # create a dict that maps these argument names together with their default value
        init_args, _, _, init_defaults, _, _, _ = getfullargspec(fn)
        init_defaults = init_defaults or () # should not be NoneType
        num_defaults = len(init_defaults)
        defaults = dict(zip(init_args[-num_defaults:], init_defaults))
        others = dict((a, None) for a in init_args[:-num_defaults])
        return {**defaults, **others}

    def dict_from_frame(frame, fn):
        args, vargs, kwargs, locals = getargvalues(frame)
        fn_args, _, _, fn_defaults, _, _, _ = getfullargspec(fn)
        fn_defaults = fn_defaults or () # should not be NoneType
        num_vargs = len(locals[vargs])
        matching_fn_args = fn_args[1:num_vargs+1] if fn_args[0] is 'self' else fn_args[:num_args] # get rid of self

        vargs_dict = dict(zip(matching_fn_args, locals[vargs]))
        args_dict = dict((a, locals[a]) for a in args)
        kwargs_dict = locals[kwargs]

        return {**args_dict, **vargs_dict, **kwargs_dict}
    
    def new_init(self, *args, **kwargs):
        # call the undecorated function
        # determine the default arguments using dict_from_fn
        # catch the supplied arguments using inspector.currentframe()
        init(self, *args, **kwargs) 
        self.__defaultInit__ = init 
        self.__defaultConfiguration__ = dict_from_fn(init)
        self.__configuration__ = dict_from_frame(currentframe(), init)
    return new_init


class Configurable(metaclass=ABCMeta):
    '''
    __new__ is called to create a new object, before invoking it's constructor.
    this allows us to modify it's constructur, i.e. automatically decorate it.
    The following code saves use from decorating as follows:
    @configurable
    def __init__(self, *args, *kwargs):
    '''
    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)#, *args, **kwargs)
        if instance.__class__.__init__.__name__ is "__init__": # only decorate when it is unmodified
            instance.__class__.__init__ = configurable(instance.__class__.__init__)
        return instance

    def configurable_properties(self):
        return self.__defaultConfiguration__.keys()

    @staticmethod
    def from_configuration(configuration, resolve_class_fn):
        if isinstance(configuration, collections.Iterable) and "self" in configuration:
            class_name = configuration['self']
            myClass = resolve_class_fn(class_name)
            args = dict((k, Configurable.from_configuration(v, resolve_class_fn)) for k,v in configuration.items() if k is not "self")
            print("new class %s with args %s" % (myClass, args))
            return myClass(**args)
        else:
            return configuration

    def get_default_configuration(self):
        if not hasattr(self, "__defaultConfiguration__"):
            raise ValueError("Did you mark your configuration function of %s @configurable?" % self)
        return self.__defaultConfiguration__

    def get_configuration(self, c: dict = None):
        def solve(k,v):
            if k is "self":
                return (k, v.__class__.__name__) # refer to object as its classname
            if k is not "self" and isinstance(v, Configurable):
                return (k, v.get_configuration())
            if k is not "self" and isinstance(v, tf.keras.models.Model):
                return (k, v.to_json())
            return (k,v)
        
        if c is None: #base case
            if not hasattr(self, "__defaultConfiguration__"):
                raise ValueError("Did you mark your configuration function of %s @configurable?" % self)

            return self.get_configuration({**self.__defaultConfiguration__, **self.__configuration__})
        else: # resolve recusrive configurations
            return dict(solve(k,v) for k,v in c.items())


