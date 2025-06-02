import com.squareup.moshi.*
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import java.io.*
import okio.*
import java.util.TreeSet

typealias AdapterProgram = Pair<JsonAdapter<BrilProgram>, BrilProgram>

fun moshi(): Moshi {
    return Moshi.Builder()
        .add(BrilPrimitiveValueTypeAdapter())
        .add(BrilTypeAdapter())
        .add(BrilInstrAdapter())
        .add(BrilOpAdapter())
        .add(BriloopInstrAdapter())
        .add(BriloopTypeAdapter())
        .add(BriloopValueAdapter())
        .addLast(KotlinJsonAdapterFactory())
        .build()
}

@kotlin.ExperimentalStdlibApi
fun brilProgram(
    args: Array<String>,
    moshi: Moshi,
): AdapterProgram? {
    val argSet: TreeSet<String> = TreeSet()
    for (i in 0 ..< args.size) {
        argSet.add(args[i]!!)
    }

    val filename = if (args.size > 1 && args[1].startsWith("-f")) {
        args[2]
    } else {
        null
    }

    val source = brilProgramSource(filename)
    val adapter = brilAdapter(moshi) 
    val program = adapter.fromJson(source)

    if (program != null) {
        return adapter to program
    } else {
        return null
    }
}

private fun brilProgramSource(filename: String?) =
    if (filename == null) {
        System.`in`.source().buffer()
    } else {
        val file = File(filename)
        val source = file.source()
        source.buffer()
    }

@kotlin.ExperimentalStdlibApi
fun brilAdapter(
    moshi: Moshi,
): JsonAdapter<BrilProgram> {
    val adapter: JsonAdapter<BrilProgram> = moshi.adapter<BrilProgram>()
    return adapter
} 


