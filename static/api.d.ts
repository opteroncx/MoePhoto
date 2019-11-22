/**
Process a single file, determine whether image or video by the file's extension.
@param filepath Path to the file to process.
@param preset Name of the preset to use for the file.
@returns Promise resolve to the processed file's path or reject if any error.
**/
type processFunction = (filepath: string, preset: string) => Promise<Object>

const log = (result: Object) => Object
type FilesHandler = (files: Array<string>) => void | Array<string>
type callbackFunction<T> = (result: Object) => T
type ProcessError = [Error, string, number]

/**
Process all images or videos (with known extension) in a given directory recursively.
@param dir Path to the directory to process.
@param preset Name of the preset to use for processing.
@param isVideo Process video or image. @default false
@param filesHandler Function to filter the files list from the directory. @default identity
@param callback Function to call every time a file is processed. @default log
@returns Promise resolve to array of callback's return value or processing errors.
**/
type processFolderFunction<T> = (dir: string, preset: string, isVideo = false, filesHandler: FilesHandler = identity, callback: callbackFunction<T> = log) => Promise<Array<T | ProcessError>>

/**
Construct MoePhoto's API.
@param url MoePhoto's servicing address. @default 'localhost'
@param port MoePhoto's listening port. @default 2333
@returns Function object {process, processFolder}.
**/
export function MoePhoto(url = 'localhost', port = 2333): { process: processFunction, processFolder: processFolderFunction }