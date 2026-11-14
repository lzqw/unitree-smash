set pagination off
set confirm off
set debuginfod enabled off
handle SIGPIPE nostop noprint pass
python
import gdb, json

def py_ascii(obj):
    header = obj.cast(gdb.lookup_type('PyASCIIObject').pointer()).dereference()
    if not int(header['state']['ascii']):
        return '<non-ascii unicode>'
    address = int(obj) + gdb.lookup_type('PyASCIIObject').sizeof
    return gdb.Value(address).cast(gdb.lookup_type('char').pointer()).string(length=int(header['length']))

class StaticOwner(gdb.Breakpoint):
    def __init__(self, base):
        super().__init__('*'+hex(base+0x1107b5), internal=True)
        self.base = base
    def stop(self):
        gdb.write('\n=== STATIC PYTHON LIST INITIALIZATION ===\n')
        gdb.execute('bt 24')
        try:
            slot=gdb.Value(self.base+0x19ae60).cast(gdb.lookup_type('PyObject').pointer().pointer())
            obj=slot.dereference()
            gdb.write('STATIC_TYPE='+obj.dereference()['ob_type'].dereference()['tp_name'].string()+'\n')
            lst=obj.cast(gdb.lookup_type('PyListObject').pointer()).dereference()
            gdb.write('STATIC_LIST_SIZE='+str(int(lst['ob_base']['ob_size']))+'\n')
            state=gdb.parse_and_eval('_PyRuntime.gilstate.tstate_current._value').cast(gdb.lookup_type('PyThreadState').pointer())
            frame=state.dereference()['cframe'].dereference()['current_frame']
            records=[]
            while int(frame) and len(records)<40:
                code=frame.dereference()['f_code'].dereference()
                records.append(dict(file=py_ascii(code['co_filename']),function=py_ascii(code['co_name']),first_line=int(code['co_firstlineno'])))
                frame=frame.dereference()['previous']
            gdb.write('PYTHON_INITIALIZATION_FRAMES='+json.dumps(records)+'\n')
        except Exception as error:
            gdb.write('OWNER_INSPECTION_ERROR='+str(error)+'\n')
        self.enabled=False
        return False

def loaded(event):
    if event.new_objfile.filename.endswith('/bindings/_physx.cpython-311-x86_64-linux-gnu.so'):
        try:
            base=int(gdb.parse_and_eval('(void*)PyInit__physx'))-0xb4360
            gdb.write('STATIC_OWNER_BREAKPOINT_BASE='+hex(base)+'\n')
            StaticOwner(base)
        except Exception as error:
            gdb.write('BREAKPOINT_SETUP_ERROR='+str(error)+'\n')
gdb.events.new_objfile.connect(loaded)
end
run
printf "\n=== STOP REASON ===\n"
p $_siginfo
bt 30
