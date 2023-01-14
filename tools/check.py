import glob
import os
import argparse

check_training_diff_file = '/root/autodl-tmp/check_training_diff_file'
check_training_file_path_src='/root/autodl-tmp/openlane/training/'
check_training_file_path_dst='/root/autodl-tmp/extend/labels/training/'

check_validation_diff_file = '/root/autodl-tmp/check_validation_diff_file'
check_validation_file_path_src='/root/autodl-tmp/openlane/validation/'
check_validation_file_path_dst='/root/autodl-tmp/extend/labels/validation/'


def file_find(start, name):
    for relpath, dirs, files in os.walk(start):
        if name in files:
            full_path = os.path.join(start, relpath, name)
            return os.path.normpath(os.path.abspath(full_path))

def string_file_add(file, context):
    with open(file,'a') as f:
        f.write(context)


def file_check(set_class):
    if set_class == 'training':
        check_file_path_src=check_training_file_path_src
        check_file_path_dst=check_training_file_path_dst
        check_diff_file=check_training_diff_file

    else:
        check_file_path_src=check_validation_file_path_src
        check_file_path_dst=check_validation_file_path_dst
        check_diff_file=check_validation_diff_file

    #clean diff file
    if os.path.exists(check_diff_file):
        os.remove(check_diff_file)

    check_label_list_src=glob.glob( check_file_path_src+ '**/*.json', recursive=True) 
    for label_file in check_label_list_src:

        (orig_path, orig_file) = os.path.split(label_file)
        (_, tf_tag) = os.path.split(orig_path)

        (filename,extension) = os.path.splitext(orig_file)
        check_file_src =filename + '.1' + extension

        #print( check_file_src)

        ref_json_file = file_find(check_file_path_dst + tf_tag, check_file_src)

        #print(tf_tag)
        
        #print(ref_json_file)
        if  ref_json_file is None:
            print(label_file)
            print(check_file_path_dst + tf_tag)
            string_file_add(check_diff_file,  label_file+'\n')


def main():
    parser = argparse.ArgumentParser(description='paper work')
    parser.add_argument('--class_set', type=str, default='training', help='The class of set')
    args = parser.parse_args() 

    if args.class_set ==  'validation' or args.class_set == 'training':
        file_check(args.class_set)
    else:
        print(args.class_set + ": error input")



if __name__ == '__main__':
    main()


