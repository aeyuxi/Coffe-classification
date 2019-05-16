from __future__ import print_function
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import os
from sklearn.preprocessing import normalize


plt.style.use('ggplot' )
img_width    = 100
img_height   = 100
# =============================================================================
#img_read 
#dirname: Path directory where the images are located 
#returns: Dictionary {img_paths, label_list}
#Images will be labeled as follow Good = 1, Bad = 0
# =============================================================================
def img_read ( filenames ):
    label_list = []
    imgs_path =[]
    
    for imgs_names in os.listdir( filenames ):
        imgs_path.append( os.path.join( filenames, imgs_names ) )
        if 'good' in imgs_names:
            label_list.append( 1 )     
        else:
            label_list.append( 0 )
                  
    return  {'imgs_paths'  : imgs_path,
             'label_array' : np.array( label_list ).astype( np.float32 )}
    
# =============================================================================
#img_norm
#imgs_path: List with the path of each image 
#returns: array of normalize and resized images
# =============================================================================
def img_norm ( imgs_path ):
    
    img_list     = []
    
    #Resize each image to img_width and img_heigth    
    for images in imgs_path:
       img = cv2.cvtColor( cv2.imread( images ), cv2.COLOR_BGR2GRAY )
       width, height = img.shape
       if width!= img_width or height!= img_height :
           img = cv2.resize( img, dsize=( img_width, img_height ), interpolation=cv2.INTER_CUBIC )
       img_list.append( img )    
    
    img_array = np.array( img_list )
    
#    TODO: Use this metrics for the normalization
#    mean_img = np.mean(img_array, axis=0 )
#    sigma_img = np.std(img_array, axis=0 )

    #each row is an image of the dataset
    dataset = np.reshape( img_array, [len( img_array ), img_height*img_width] )
    norm_dataset = normalize( dataset.astype( np.float32 ) , norm='max' )
    return norm_dataset
#    norm_dataset = normalize((dataset.astype(np.float32) - mean_img)/sigma_img)


# =============================================================================
# Accuracy calculation
#h: The hypotesis, the predicte values 
#dataset_array: The tranning or testing dataset of images 
#dataset_label: The labels of each image of the dataset     
# =============================================================================
def accuracy( h, dataset_array, dataset_label_array ):
      predictions = h.eval( feed_dict = {X: dataset_array} )
      result = ( np.where( predictions>=0.5,1,0 ) )
      accuracy_array = abs( np.reshape( result,[len( dataset_array ),] ) - dataset_label_array )
      accuracy_list = list( accuracy_array )
      accuracy_float = float( accuracy_list.count( 0.0 )/len( dataset_array) )
      return accuracy_float 
      
    

# =============================================================================
# Model definition for Linear Regression (X*W + b) with sigmoid function
#TODO: search which is the best loos function and non-linearity     
# =============================================================================
try:
    n_output = 1
    n_features = img_width * img_height      
    layer_dim_list = [n_features, n_output]
    
    X = tf.placeholder( dtype=tf.float32,
                        shape = [None, n_features], 
                        name = 'input')
    
    Y = tf.placeholder( dtype=tf.float32,
                       shape = [n_output], 
                       name = 'output')
    
    current_input = X        
    Ws_list =[]
    
    for i, layer_i in enumerate(layer_dim_list):    
        with tf.variable_scope("percpetron/{}".format( int( layer_i ) ) ):
            n_input = int( layer_dim_list[i] )
            if i + 1 < len( layer_dim_list ):
                n_output = int(layer_dim_list[i + 1])
            else:
                break
            
            W = tf.get_variable( name = 'W',
                                 dtype=tf.float32,
                                 shape = [n_input, n_output],
                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            
            b = tf.get_variable( name = 'b',
                                 shape = [n_output],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer( 0.01 ) )
            
            
            h = tf.nn.bias_add( name='h',
                                value=tf.matmul(current_input, W),
                                bias=b)
        
            current_input = tf.nn.relu( h ) 
        
            Ws_list.append(W)
        
    pred = current_input    
    cost = tf.reduce_mean( tf.square( h - Y ) )
    optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

    init = tf.global_variables_initializer( )
   
    
    with tf.Session( ) as sess:
        sess.run( init )
        cwd = os.getcwd( )

        dir_coffe_trainning = 'coffe_beans_trainning'
        filenames_trainning = os.path.join( cwd, dir_coffe_trainning )
        dataset_dict_trainning  = img_read( filenames_trainning )
        imgs_path_trainning     = dataset_dict_trainning['imgs_paths']
        label_array_trainning   = dataset_dict_trainning['label_array']
        dataset_array_trainning = img_norm( imgs_path_trainning )
        
        dir_coffe_testing = 'coffe_beans_testing'
        filenames_testing = os.path.join( cwd, dir_coffe_testing )        
        dataset_dict_testing  = img_read( filenames_testing )
        imgs_path_testing     = dataset_dict_testing['imgs_paths']
        label_array_testing   = dataset_dict_testing['label_array']
        dataset_array_testing = img_norm( imgs_path_testing )
        
        
        avg_cost_list = []
        acc_tr_list   = []
        acc_ts_list   = []
        n_iterations = 20
        for it_i in range( n_iterations ):

            cost_pred_ac = 0
            for ( img, y ) in zip( dataset_array_trainning, label_array_trainning ):
                img_input = np.reshape( img,[1,img_width*img_height] )
                y_output  = np.reshape( y,[1,] )
                _ , cost_pred, val_pred = sess.run( [optimizer, cost, pred],feed_dict={X: img_input, Y :  y_output } )                           
                cost_pred_ac += cost_pred
          
            avg_cost_pred =  cost_pred_ac / len( dataset_array_trainning )
            print( 'Epoch :  %s    cost :  %s  '  %( it_i, avg_cost_pred ) )
            avg_cost_list.append( avg_cost_pred )
        
            accuracy_trainning = accuracy( pred, dataset_array_trainning, label_array_trainning )
            acc_tr_list.append(accuracy_trainning)
            print( 'Accuracy trainning' , accuracy_trainning )
            
            
            accuracy_testing = accuracy( pred, dataset_array_testing, label_array_testing )
            acc_ts_list.append(accuracy_testing)
            print( 'Accuracy testing' , accuracy_testing )
            
            
        
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        
        axs[0].plot(avg_cost_list)
        axs[0].set_title('Cost Function')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Cost')
        
        axs[1].plot(acc_tr_list)
        axs[1].set_title('Accuracy Trainning')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        
        axs[2].plot(acc_ts_list)
        axs[2].set_title('Accuracy Testing')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Accuracy')
                
        
        
#    while True:
#        key = cv2.waitKey( 1 )
#        cv2.imshow( 'mean', mean_img.astype( np.uint8 ) )
#        cv2.imshow( 'std', sigma_img.astype( np.uint8 ) )
#        if key == ord( 'q' ):
#            cv2.destroyAllWindows(  )   
#            tf.reset_default_graph(  )
#            break

except tf.errors.InvalidArgumentError as e:
   print( e )

finally:
    cv2.destroyAllWindows( ) 
    tf.reset_default_graph(  )

# =============================================================================

