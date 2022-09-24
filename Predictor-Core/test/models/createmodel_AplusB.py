import numpy as np

import tensorflow
version_TF = tensorflow.__version__



def example_pb():

    if version_TF.startswith('2'):
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
    elif version_TF.startswith('1'):
        import tensorflow as tf

    # TF1 user please just use:
    # import tensorflow as tf

    # Two simple inputs
    a = tf.placeholder(tf.float32, shape=(None, 2), name="input_a")
    b = tf.placeholder(tf.float32, shape=(None, 2), name="input_b")

    # Output
    c = tf.add(a, b, name='result')

    # Try to run some data
    data_a = np.reshape([1, 2, 3, 4, 5, 6], (-1, 2))
    data_b = np.reshape([6, 5, 4, 3, 2, 1], (-1, 2))

    # Run the model to get result
    print("Running A + B ...")
    sess = tf.Session()
    data_c = sess.run(c, feed_dict = {a:data_a, b:data_b})

    print("Result:\n", np.squeeze(data_a), ' + ', np.squeeze(data_b), ' = ', np.squeeze(data_c))

    # Save the model
    print("Saving model A + B ...")

    if version_TF.startswith('2'):
        with open('simple_graph_tf2.pb', 'wb') as f:
            f.write(tf.get_default_graph().as_graph_def().SerializeToString())

    elif version_TF.startswith('1'):
        with open('simple_graph_tf1.pb', 'wb') as f:
            f.write(tf.get_default_graph().as_graph_def().SerializeToString())




def example_SavedModel():

    # TF2 user:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # TF1 user please just use:
    # import tensorflow as tf

    # Two simple inputs
    input_a = keras.layers.Input(shape=(2,), name="input_a")
    input_b = keras.layers.Input(shape=(2,), name="input_b")

    # Output
    added = keras.layers.Add(name="result")([input_a, input_b])

    # Build model
    model = keras.models.Model(inputs=[input_a, input_b], outputs=added)

    # Try to run some data
    data_a = np.reshape([1, 2, 3, 4, 5, 6], (-1, 2))
    data_b = np.reshape([6, 5, 4, 3, 2, 1], (-1, 2))

    # Run the model to get result
    print("Running A + B ...")
    data_c = model.predict([data_a, data_b])
    print("Result:\n", np.squeeze(data_a), ' + ',
          np.squeeze(data_b), ' = ', np.squeeze(data_c))

    print("Saving model A + B ...")

    if version_TF.startswith('2'):
        model.save("saved_tf2_model", save_format="tf")
    elif version_TF.startswith('1'):
        from tensorflow.keras import backend as K
        from tensorflow.saved_model.signature_def_utils import predict_signature_def
        from tensorflow.saved_model import tag_constants
        sess = K.get_session()
        builder = tf.saved_model.builder.SavedModelBuilder("./saved_tf1_model")
        signature = predict_signature_def(inputs={"input_a": input_a, "input_b": input_b},
                                          outputs={"result": added})  
        builder.add_meta_graph_and_variables(sess=sess,
                                            tags=[tag_constants.SERVING],
                                            signature_def_map={'predict': signature})
        builder.save()

        
        
    




if __name__ == "__main__":
    import sys
    arg_v = sys.argv
    print("TF version is: \n", version_TF)
    if len(arg_v) == 1:
        print("Please enter the model type you want to save:\n", "PB or SavedModel\n")
        print("Using default pb format")
        example_pb()
    else:
        mode = arg_v[1]
        if mode == "PB":
            example_pb()
        elif mode == "SavedModel":
            example_SavedModel()
        else:
            print("Unknown save model options, please use: \n",
                "python createmodel_AplusB.py PB \n",
                "Or \n", "python createmodel_AplusB.py SavedModel ")






