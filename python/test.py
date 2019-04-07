def syntaxDemo():
    x = 1

    x += 2
    print(x)

    y = 'Hello '
    y1 = 'world'
    print(y + y1)

    z1, z2 = 1, 'str'
    print(z1, z2)

    '''
    swap z1 and z2
        temp = z1
        z1 = z2
        z2 = temp
    '''
    z1, z2 = z2, z1
    print(z1, z2)

def datatypeDemo():

    list_sample = [1, 2]
    print(list_sample)
    list_sample1 = ['hello', 'world']
    print(list_sample1)

    tuple_sample = (1, 2, 3)
    print(tuple_sample)
    tuple_sample = ('hello', 'world')
    print(tuple_sample)


    dict_sample = {'key1': 'value1', 2:3, 'tuple': (1, 2)}
    print(dict_sample['key1'])
    print(dict_sample['tuple'][0])


    mix_sample = [5, 'hello world', [1, 'alex', (1, 2, 3)], ('my', 'tuple')]
    print(mix_sample[0])
    print(mix_sample[-1])


    function_sample = type
    print(function_sample(list_sample))
    print(type(list_sample))


    list_sample2 = [1, 'a', 2, 'b', 3, 'c']

    first, *_ = list_sample2
    print(first)

    *_, last = list_sample2
    print(last)

    _, second, *_ = list_sample2
    print(second)

def stringDemo():

    print('Name: %s' %  ('alex'))

    print('Nmae: %(name)s' % {'name':'alex'})

    print('Name: {}'.format('alex'))

    multi_string = '''this is a
    multiline
    string'''
    print(multi_string)

    print('Name: %s\nNumber: %d\nString: %s\n' % ('alex', 1, 4 * '-'))

    print('this %(value1)s a %(value2)s\n' % {'value2':'test', 'value1':'is'})

    message = 'let\'s learn {} in a {}'.format('python', 'short time')

    print(message)

def flowControlDemo():
    range_list = range(10)
    '''
    it will be written if we use c or c++
    for (i = 0; i < 10; i++) {
        range_list[i] = i;
    }
    '''
    print(range_list)

    for i in range_list:
        print(i, end='')

    for number in range_list:
        if number in (3, 4, 7, 9):
            break
        else:
            continue
    else:
        pass

    if range_list[1] == 2:
        print("The second item (lists are 0-based) is 2")
    elif range_list[1] == 3:
        print("The second item (lists are 0-based) is 3")
    else:
        print("\nDunno")

    while range_list[0] == 1:
        pass

    odd = 0
    even = 0
    for number in range(100):
        if number % 2 == 0:
            even += number
        else:
            odd += number

    print('the total of odd:', odd)
    print("the total of even:", even)

def functionDemo():
    class People:
        '''
        Base class. It has two attributes: name and age,
        and two methods: get_name() and get_age()j to obtain attribute value.
        When declaring classes, you must rewrite __init__ method to initialize
        class for most time.
        '''

        def __init__(self, name, age):
            "Initialization"
            self.name = name
            self.age = age

        def get_age(self):
            return self.age

        def get_name(self):
            return self.name

    people = People(name='XiaoMing', age=21)
    print(people)

    # using class's method to get its attribute value
    print('age :', people.get_age())
    print('name :', people.get_name())


    print('age :', people.age)
    print('name :', people.name)

    class Student(People):
        '''
        '''
        def __init__(self, name, age, idx):
            super(Student,self).__init__(name, age)
            self.idx = idx

        def get_idx(self):
            return self.idx

    student = Student('XiaoMing', 21, 12)
    print(student)

    print('age :', student.get_age())
    print('name :', student.get_name())
    print('idx :', student.get_idx())

    print('age :', student.age)
    print('name :', student.name)
    print('idx :', student.idx)

def fileIODemo():
    with open("./example.txt", 'w') as f:
        f.writelines("hello world!")
        f.writelines('Bye bye!')

    with open("./example.txt", 'r') as f:
        print(f.readlines())

def miscellaneousDemo():
    list1 = [1, 2, 3]
    list2 = [3, 4, 5]
    print([x * y for x in list1 for y in list2])

    print([x for x in list1 if 1 < x < 4])

    print(any(i % 3 for i in range(10)))

    print(sum(1 for i in [2,2,3,3,4,4] if i == 4))



if __name__ == "__main__":
    miscellaneousDemo()
