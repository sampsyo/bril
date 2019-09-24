/**
 * An abstract key class used to access the heap.
 * This allows for "pointer arithmetic" on keys,
 * while still allowing lookups based on the based pointer of each allocation.
 */
export class Key {
    readonly base: number;
    readonly offset: number;

    constructor(b:number, o:number) {
        this.base = b;
        this.offset = o;
    }

    add(offset:number) {
        return new Key(this.base, this.offset + offset);
    }
}

/**
 * 
 */
export class Heap<X> {

    private readonly storage: Map<number, X[]>
    constructor() {
        this.storage = new Map()
    }

    isEmpty(): boolean {
        return this.storage.size == 0;
    }

    private count = 0;
    private getNewBase():number {
        let val = this.count;
        this.count++;
        return val;
    }

    private freeKey(key:Key) {
        return;
    }

    alloc(amt:number): Key {
        if (amt <= 0) {
            throw `cannot allocate ${amt} entries`
        }
        let base = this.getNewBase();
        this.storage.set(base, new Array(amt))
        return new Key(base, 0);
    }

    free(key: Key) {
        if (this.storage.has(key.base) && key.offset == 0) {
            this.freeKey(key);
            this.storage.delete(key.base);
        } else {
            throw `Tried to free illegal memory location base: ${key.base}, offset: ${key.offset}. Offset must be 0.`
        }
    }

    write(key: Key, val:X) {
        let data = this.storage.get(key.base);
        if (data && data.length > key.offset && key.offset >= 0) {
            data[key.offset] = val;
        } else {
            throw `Uninitialized heap location ${key.base} and/or illegal offset ${key.offset}`
        }
    }

    read(key: Key):X {
        let data = this.storage.get(key.base);
        if (data && data.length > key.offset && key.offset >= 0) {
            return data[key.offset];
        } else {
            throw `Uninitialized heap location ${key.base} and/or illegal offset ${key.offset}`
        }
    }
}